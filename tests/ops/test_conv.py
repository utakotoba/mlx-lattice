from __future__ import annotations

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)
from mlx_lattice.ops._relation_exec import (
    sparse_conv_features_sorted_direct_reference_from_relation,
)
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    assert_same_sparse_identity,
    mx,
)

pytestmark = [
    pytest.mark.ops,
    pytest.mark.conv,
    pytest.mark.usefixtures('selected_backend'),
]


def test_conv3d_pointwise_matches_dense_linear_contract() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = conv3d(x, weight, bias, kernel_size=1)

    assert out.feats.tolist() == [[9.0, 18.0], [19.0, 42.0]]
    assert_same_sparse_identity(out, x)


def test_conv3d_generic_matches_fused_native_reference() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(x, weight, kernel_size=(3, 1, 1))

    assert active_coords(out) == coords.tolist()
    assert active_feats(out).tolist() == [[8.0], [14.0], [8.0]]
    assert out.stride == (1, 1, 1)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


def test_conv3d_generic_supports_float16() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float16)
    x = SparseTensor(coords, feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float16).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(x, weight, kernel_size=(3, 1, 1))
    mx.eval(out.feats)

    assert out.feats.dtype == mx.float16
    assert_nested_close(
        active_feats(out).astype(mx.float32).tolist(),
        [[8.0], [14.0], [8.0]],
    )


def test_sorted_implicit_gemm_direct_reference_matches_classic(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('sorted direct row-stationary reference is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(96)],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [
                ((row + 1) * (channel + 3) % 17) / 17.0
                for channel in range(32)
            ]
            for row in range(96)
        ],
        dtype=mx.float16,
    )
    x = SparseTensor(coords, feats)
    weight = mx.array(
        [((index % 23) - 11) / 23.0 for index in range(32 * 27 * 32)],
        dtype=mx.float16,
    ).reshape((32, 3, 3, 3, 32))
    relation = x.coord_manager.kernel_relation(x.coord_key, kernel_size=3)
    automatic = conv3d(x, weight, kernel_size=3).feats
    direct = sparse_conv_features_sorted_direct_reference_from_relation(
        x.feats,
        weight,
        relation,
        store_sorted=True,
    )
    reorder_rows = relation.require_sorted_implicit_gemm().reorder_rows
    mx.eval(automatic, direct, reorder_rows)

    sorted_automatic = automatic[reorder_rows].astype(mx.float32)
    assert float(mx.max(mx.abs(sorted_automatic - direct)).item()) <= 0.003


def test_sorted_implicit_gemm_preserves_convolution_autodiff_contract(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('sorted implicit GEMM is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(96)],
        dtype=mx.int32,
    )
    feats = mx.ones((96, 32), dtype=mx.float16)
    weight = mx.ones((32, 3, 3, 3, 32), dtype=mx.float16) / 32

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

    gradients = mx.grad(loss, argnums=(0, 1))(feats, weight)
    mx.eval(*gradients)

    assert gradients[0].shape == feats.shape
    assert gradients[1].shape == weight.shape
    assert gradients[0].dtype == mx.float16
    assert gradients[1].dtype == mx.float16


def test_conv3d_automatic_dispatch_falls_back_for_unsupported_igemm_shape(
    selected_backend,
) -> None:
    if selected_backend.name != 'metal':
        pytest.skip('automatic Metal dispatch policy is Metal-only')
    coords = mx.array(
        [[0, index, 0, 0] for index in range(32)],
        dtype=mx.int32,
    )
    feats = mx.ones((32, 16), dtype=mx.float16)
    weight = mx.ones((16, 3, 3, 3, 16), dtype=mx.float16) / 16
    x = SparseTensor(coords, feats)

    out = conv3d(x, weight, kernel_size=3)
    mx.eval(out.feats)

    assert out.feats.shape == (32, 16)
    assert out.feats.dtype == mx.float16


def test_conv3d_target_coordinates_match_sparse_reference() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    target_tensor = SparseTensor(
        target,
        mx.zeros((2, 1), dtype=mx.float32),
        coord_manager=x.coord_manager,
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        coordinates=target_tensor,
    )

    assert active_coords(out) == target.tolist()
    assert active_feats(out).tolist() == [[14.0], [3.0]]
    assert out.stride == x.stride
    assert out.coord_manager is x.coord_manager
    assert out.coord_key == target_tensor.coord_key


def test_conv3d_target_same_reuses_input_coordinate_identity() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    target = SparseTensor(
        coords,
        mx.zeros_like(feats),
        coord_manager=x.coord_manager,
    )
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1, 3, 1, 1, 1
    )

    out = conv3d(
        x,
        weight,
        kernel_size=(3, 1, 1),
        coordinates=target,
    )

    assert target.coord_key == x.coord_key
    assert out.coord_key == x.coord_key
    assert active_feats(out).tolist() == [[8.0], [14.0], [8.0]]


def test_conv3d_pointwise_target_coordinates_use_sparse_relation() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([[2.0]], dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=1, coordinates=target)

    assert active_coords(out) == target.tolist()
    assert active_feats(out).tolist() == [[4.0], [0.0]]


def test_conv3d_target_path_is_autogradable() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(
            conv3d(
                x,
                weight_arg,
                kernel_size=(3, 1, 1),
                coordinates=target,
            ).feats
        )

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)

    assert grad_feats.tolist() == [[1.0], [2.0], [4.0]]
    assert grad_weight.tolist() == [[[[[4.0]]], [[[2.0]]], [[[3.0]]]]]


def test_conv3d_generic_path_is_autogradable_for_features_and_weights() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats)

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)

    assert grad_feats.tolist() == [[3.0], [6.0], [5.0]]
    assert grad_weight.tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]


def test_convolution_modes_are_autogradable_for_features_and_weights() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )

    def subm_loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(
            subm_conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats
        )

    subm_grad_feats, subm_grad_weight = mx.grad(
        subm_loss,
        argnums=(0, 1),
    )(feats, weight)

    assert subm_grad_feats.tolist() == [[3.0], [6.0], [5.0]]
    assert subm_grad_weight.tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]

    transpose_coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    transpose_feats = mx.array([[4.0]], dtype=mx.float32)
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1,
        2,
        1,
        1,
        1,
    )

    def transpose_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    def generative_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            generative_conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    expected_weight_grad = [[[[[4.0]]], [[[4.0]]]]]
    for loss in (transpose_loss, generative_loss):
        grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(
            transpose_feats,
            transpose_weight,
        )
        assert grad_feats.tolist() == [[5.0]]
        assert grad_weight.tolist() == expected_weight_grad


def test_conv3d_generic_supports_explicit_vjp_and_jvp_transforms() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    tangent = mx.ones_like(feats)
    weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        1,
        3,
        1,
        1,
        1,
    )
    weight_tangent = mx.ones_like(weight)

    def features(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats

    outputs, grads = mx.vjp(
        features,
        [feats, weight],
        [mx.ones((3, 1), dtype=mx.float32)],
    )
    _, jvps = mx.jvp(
        features,
        [feats, weight],
        [tangent, weight_tangent],
    )

    assert outputs[0].tolist() == [[8.0], [14.0], [8.0]]
    assert grads[0].tolist() == [[3.0], [6.0], [5.0]]
    assert grads[1].tolist() == [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]]
    assert jvps[0].tolist() == [[8.0], [12.0], [8.0]]


def test_convolution_modes_are_compatible_with_mx_compile(
    compile_backend,
) -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    def generic(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        return conv3d(
            SparseTensor(coords, feats_arg),
            weight_arg,
            kernel_size=(3, 1, 1),
        ).feats

    def subm(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        return subm_conv3d(
            SparseTensor(coords, feats_arg),
            weight_arg,
            kernel_size=(3, 1, 1),
        ).feats

    for fn in (generic, subm):
        compiled = mx.compile(fn)
        assert compiled(feats, weight).tolist() == [[3.0], [6.0], [5.0]]

    transpose_coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    transpose_feats = mx.array([[4.0]], dtype=mx.float32)
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1,
        2,
        1,
        1,
        1,
    )

    def transposed(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return conv_transpose3d(
            x,
            weight_arg,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        ).feats

    def generated(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return generative_conv_transpose3d(
            x,
            weight_arg,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        ).feats

    for fn in (transposed, generated):
        compiled = mx.compile(fn)
        assert compiled(
            transpose_feats,
            transpose_weight,
        ).tolist() == [[8.0], [12.0]]


def test_conv3d_strided_updates_output_stride_and_coordinates() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=1, stride=2)

    assert active_coords(out) == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert active_feats(out).tolist() == [[1.0], [3.0]]
    assert out.stride == (2, 2, 2)


def test_subm_conv3d_reuses_input_coordinate_identity() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    out = subm_conv3d(x, weight, kernel_size=(3, 1, 1))

    assert out.feats.tolist() == [[3.0], [6.0], [5.0]]
    assert_same_sparse_identity(out, x)


def test_transpose_convs_generate_the_same_output_contract() -> None:
    x = SparseTensor(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[4.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(1, 2, 1, 1, 1)

    out = conv_transpose3d(
        x,
        weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    generated = generative_conv_transpose3d(
        x,
        weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )

    assert active_coords(out) == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert active_feats(out).tolist() == [[8.0], [12.0]]
    assert out.stride == (1, 1, 1)
    assert active_coords(generated) == active_coords(out)
    assert active_feats(generated).tolist() == active_feats(out).tolist()
    assert generated.stride == out.stride


def test_conv_ops_reject_ambiguous_contracts() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
    )

    with pytest.raises(ValueError, match='odd kernel_size'):
        subm_conv3d(x, mx.ones((2, 1, 1), dtype=mx.float32), kernel_size=2)

    with pytest.raises(ValueError, match='must divide'):
        conv_transpose3d(x, mx.ones((2, 1, 1), dtype=mx.float32))
