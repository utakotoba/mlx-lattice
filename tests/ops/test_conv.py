from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    assert_same_sparse_identity,
    mx,
    run_with_gpu_default,
    skip_without_metal,
)


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


@pytest.mark.parametrize(
    ('channels_in', 'channels_out', 'dtype'),
    [
        (16, 32, mx.float16),
        (32, 16, mx.float16),
        (64, 32, mx.float32),
        (64, 64, mx.float16),
    ],
)
def test_conv3d_generic_supports_dense_channel_pairs_on_metal(
    channels_in: int,
    channels_out: int,
    dtype: mx.Dtype,
) -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [
            [
                ((row + channel) % 7 - 3) / 7.0
                for channel in range(channels_in)
            ]
            for row in range(3)
        ],
        dtype=dtype,
    )
    weight = mx.array(
        [
            ((index % 13) - 6) / 13.0
            for index in range(channels_out * 3 * channels_in)
        ],
        dtype=dtype,
    ).reshape(channels_out, 3, 1, 1, channels_in)
    x = SparseTensor(coords, feats)

    expected = conv3d(x, weight, kernel_size=(3, 1, 1)).feats
    mx.eval(expected)

    def run() -> list[list[float]]:
        out = conv3d(x, weight, kernel_size=(3, 1, 1))
        mx.eval(out.feats)
        assert out.feats.shape == (3, channels_out)
        assert out.feats.dtype == dtype
        return active_feats(out).astype(mx.float32).tolist()

    assert_nested_close(
        run_with_gpu_default(run),
        expected.astype(mx.float32).tolist(),
        abs=2e-2 if dtype == mx.float16 else 1e-4,
    )


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


def test_conv3d_generic_weight_grad_matches_cpu_at_tensor_ops_boundary() -> (
    None
):
    skip_without_metal()
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [((row + 1) * (channel + 3) % 37) / 37.0 for channel in range(16)]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0 for index in range(16 * 3 * 3 * 3 * 16)
    ]

    def weight_grad() -> list[object]:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            16, 3, 3, 3, 16
        )

        def loss(weight_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats)
            return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

        grad = mx.grad(loss)(weight)
        mx.eval(grad)
        return grad.tolist()

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        expected = weight_grad()
    finally:
        mx.set_default_device(previous)
    actual = run_with_gpu_default(weight_grad)

    assert_nested_close(actual, expected, abs=2e-2)


@pytest.mark.parametrize('channels', [32, 64])
def test_conv3d_generic_weight_grad_tensor_ops_blocks_match_cpu(
    channels: int,
) -> None:
    skip_without_metal()
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [
            ((row + 1) * (channel + 3) % 37) / 37.0
            for channel in range(channels)
        ]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0
        for index in range(channels * 3 * 3 * 3 * channels)
    ]

    def weight_grad() -> list[object]:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            channels, 3, 3, 3, channels
        )

        def loss(weight_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats)
            return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

        grad = mx.grad(loss)(weight)
        mx.eval(grad)
        return grad.tolist()

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        expected = weight_grad()
    finally:
        mx.set_default_device(previous)
    actual = run_with_gpu_default(weight_grad)

    assert_nested_close(actual, expected, abs=2e-2)


@pytest.mark.parametrize(
    ('channels_in', 'channels_out'),
    [(16, 64), (64, 16)],
)
def test_conv3d_generic_asymmetric_backward_matches_cpu(
    channels_in: int,
    channels_out: int,
) -> None:
    skip_without_metal()
    rows = 4096
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [
            ((row + 1) * (channel + 3) % 37) / 37.0
            for channel in range(channels_in)
        ]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0
        for index in range(channels_out * 3 * 3 * 3 * channels_in)
    ]

    def grads() -> tuple[list[object], list[object]]:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            channels_out, 3, 3, 3, channels_in
        )

        def loss(
            feats_arg: mx.array,
            weight_arg: mx.array,
        ) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

        grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(
            feats,
            weight,
        )
        mx.eval(grad_feats, grad_weight)
        return grad_feats.tolist(), grad_weight.tolist()

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        expected = grads()
    finally:
        mx.set_default_device(previous)
    actual = run_with_gpu_default(grads)

    assert_nested_close(actual, expected, abs=2e-2)


def test_conv3d_generic_forward_matches_cpu_at_large_relation_boundary() -> (
    None
):
    skip_without_metal()
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [((row + 1) * (channel + 3) % 37) / 37.0 for channel in range(16)]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0 for index in range(16 * 3 * 3 * 3 * 16)
    ]

    def forward() -> list[Any]:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            16, 3, 3, 3, 16
        )
        x = SparseTensor(coords, feats)
        out = conv3d(x, weight, kernel_size=3).feats
        mx.eval(out)
        return cast('list[Any]', out.tolist())

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        expected = forward()
    finally:
        mx.set_default_device(previous)
    actual = run_with_gpu_default(forward)

    assert_nested_close(actual, expected, abs=2e-2)


def test_conv3d_generic_input_grad_matches_cpu_at_tensor_ops_boundary() -> (
    None
):
    skip_without_metal()
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [((row + 1) * (channel + 3) % 37) / 37.0 for channel in range(16)]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0 for index in range(16 * 3 * 3 * 3 * 16)
    ]

    def input_grad() -> list[object]:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            16, 3, 3, 3, 16
        )

        def loss(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(conv3d(x, weight, kernel_size=3).feats)

        grad = mx.grad(loss)(feats)
        mx.eval(grad)
        return grad.tolist()

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        expected = input_grad()
    finally:
        mx.set_default_device(previous)
    actual = run_with_gpu_default(input_grad)

    assert_nested_close(actual, expected, abs=2e-2)


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


def test_convolution_modes_are_compatible_with_mx_compile() -> None:
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


def test_subm_conv3d_consumes_lazy_gpu_default_weight() -> None:
    def run() -> None:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
        x = SparseTensor(coords, feats)
        weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

        out = subm_conv3d(x, weight, kernel_size=(3, 1, 1))

        assert active_feats(out).tolist() == [[3.0], [6.0], [5.0]]

    run_with_gpu_default(run)


def test_metal_convolution_modes_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[int]],
        list[list[float]],
        list[list[int]],
        list[list[float]],
    ]:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
        x = SparseTensor(coords, feats)
        weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
            1, 3, 1, 1, 1
        )

        generic = conv3d(x, weight, kernel_size=(3, 1, 1))
        subm = subm_conv3d(x, mx.ones_like(weight), kernel_size=(3, 1, 1))

        transposed_input = SparseTensor(
            mx.array([[0, 1, 0, 0]], dtype=mx.int32),
            mx.array([[4.0]], dtype=mx.float32),
            stride=(2, 1, 1),
        )
        transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
            1, 2, 1, 1, 1
        )
        transposed = conv_transpose3d(
            transposed_input,
            transpose_weight,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        generated = generative_conv_transpose3d(
            transposed_input,
            transpose_weight,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        mx.eval(
            generic.coords,
            generic.feats,
            generic.active_rows,
            subm.feats,
            transposed.coords,
            transposed.feats,
            generated.coords,
            generated.feats,
        )
        return (
            cast('list[list[float]]', active_feats(generic).tolist()),
            cast('list[list[float]]', subm.feats.tolist()),
            active_coords(transposed),
            cast('list[list[float]]', active_feats(transposed).tolist()),
            active_coords(generated),
            cast('list[list[float]]', active_feats(generated).tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[8.0], [14.0], [8.0]],
        [[3.0], [6.0], [5.0]],
        [[0, 2, 0, 0], [0, 3, 0, 0]],
        [[8.0], [12.0]],
        [[0, 2, 0, 0], [0, 3, 0, 0]],
        [[8.0], [12.0]],
    )


def test_metal_convolution_gradients_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[list[list[float]], object]:
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
            return mx.sum(
                conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats
            )

        grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(
            feats, weight
        )
        mx.eval(grad_feats, grad_weight)
        return grad_feats.tolist(), grad_weight.tolist()

    assert run_with_gpu_default(run) == (
        [[3.0], [6.0], [5.0]],
        [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]],
    )


def test_metal_convolution_mode_gradients_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[float]],
        object,
        list[list[float]],
        object,
        list[list[float]],
        object,
    ]:
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

        def subm_loss(
            feats_arg: mx.array,
            weight_arg: mx.array,
        ) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(
                subm_conv3d(
                    x,
                    weight_arg,
                    kernel_size=(3, 1, 1),
                ).feats
            )

        subm_grad_feats, subm_grad_weight = mx.grad(
            subm_loss,
            argnums=(0, 1),
        )(feats, weight)

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

        transpose_grad_feats, transpose_grad_weight = mx.grad(
            transpose_loss,
            argnums=(0, 1),
        )(transpose_feats, transpose_weight)
        generative_grad_feats, generative_grad_weight = mx.grad(
            generative_loss,
            argnums=(0, 1),
        )(transpose_feats, transpose_weight)
        mx.eval(
            subm_grad_feats,
            subm_grad_weight,
            transpose_grad_feats,
            transpose_grad_weight,
            generative_grad_feats,
            generative_grad_weight,
        )
        return (
            subm_grad_feats.tolist(),
            subm_grad_weight.tolist(),
            transpose_grad_feats.tolist(),
            transpose_grad_weight.tolist(),
            generative_grad_feats.tolist(),
            generative_grad_weight.tolist(),
        )

    assert run_with_gpu_default(run) == (
        [[3.0], [6.0], [5.0]],
        [[[[[3.0]]], [[[6.0]]], [[[5.0]]]]],
        [[5.0]],
        [[[[[4.0]]], [[[4.0]]]]],
        [[5.0]],
        [[[[[4.0]]], [[[4.0]]]]],
    )


def test_metal_convolution_jvp_matches_cpu_contract_when_available() -> (
    None
):
    def run() -> list[list[float]]:
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

        def features(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return conv3d(x, weight, kernel_size=(3, 1, 1)).feats

        _, jvps = mx.jvp(features, [feats], [tangent])
        mx.eval(jvps[0])
        return jvps[0].tolist()

    assert run_with_gpu_default(run) == [[5.0], [6.0], [3.0]]


def test_metal_convolution_respects_active_rows_capacity_contract() -> None:
    def run() -> tuple[list[list[int]], list[list[float]], list[int]]:
        coords = mx.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 99, 0, 0],
                [0, 100, 0, 0],
            ],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [100.0], [200.0]], dtype=mx.float32)
        x = SparseTensor(
            coords,
            feats,
            active_rows=mx.array([2], dtype=mx.int32),
        )
        weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

        out = conv3d(x, weight, kernel_size=(3, 1, 1))
        mx.eval(out.coords, out.feats, out.active_rows)
        return (
            active_coords(out),
            cast('list[list[float]]', active_feats(out).tolist()),
            cast('list[int]', out.active_rows.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        [[3.0], [3.0]],
        [2],
    )


def test_metal_convolution_supports_float16() -> None:
    skip_without_metal()

    def run() -> list[list[float]]:
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
        return active_feats(out).astype(mx.float32).tolist()

    assert_nested_close(run_with_gpu_default(run), [[8.0], [14.0], [8.0]])


def test_metal_convolution_rejects_unsupported_coord_dtype() -> None:
    def run() -> None:
        x = SparseTensor(
            mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int64),
            mx.array([[1.0], [2.0]], dtype=mx.float32),
        )
        weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)
        with pytest.raises(ValueError, match='Metal sparse convolution'):
            conv3d(x, weight, kernel_size=(3, 1, 1))

    run_with_gpu_default(run)


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
