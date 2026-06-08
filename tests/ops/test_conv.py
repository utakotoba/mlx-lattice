from __future__ import annotations

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    build_kernel_relation,
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)
from mlx_lattice.ops._exec import execute_spmm
from tests.support import (
    active_coords,
    active_feats,
    assert_same_sparse_identity,
    mx,
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


def test_conv3d_generic_matches_native_edge_spmm_reference() -> None:
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
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
    native_weight = mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
        3, 1, 1
    )

    assert active_coords(out) == coords.tolist()
    assert (
        active_feats(out).tolist()
        == execute_spmm(feats, native_weight, relation).tolist()
    )
    assert active_feats(out).tolist() == [[8.0], [14.0], [8.0]]
    assert out.stride == (1, 1, 1)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


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
