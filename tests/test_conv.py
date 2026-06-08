from __future__ import annotations

# ruff: noqa: E402, I001

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)


def test_conv3d_pointwise_reuses_coordinate_identity() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = conv3d(x, weight, bias, kernel_size=1)

    assert out.feats.tolist() == [[9.0, 18.0], [19.0, 42.0]]
    assert out.coord_key == x.coord_key
    assert out.coord_manager is x.coord_manager
    assert out.coords is x.coords


def test_conv3d_generic_uses_kernel_map_edges() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=(3, 1, 1))

    assert out.coords.tolist() == coords.tolist()
    assert out.feats.tolist() == [[3.0], [6.0], [5.0]]
    assert out.stride == (1, 1, 1)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


def test_conv3d_strided_updates_output_stride_and_coords() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=1, stride=2)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert out.feats.tolist() == [[1.0], [3.0]]
    assert out.stride == (2, 2, 2)


def test_subm_conv3d_reuses_input_coordinates() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)

    out = subm_conv3d(x, weight, kernel_size=(3, 1, 1))

    assert out.feats.tolist() == [[3.0], [6.0], [5.0]]
    assert out.coord_key == x.coord_key
    assert out.coords is x.coords


def test_transpose_convs_use_generated_output_coords() -> None:
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

    assert out.coords.tolist() == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert out.feats.tolist() == [[8.0], [12.0]]
    assert out.stride == (1, 1, 1)
    assert generated.coords.tolist() == out.coords.tolist()
    assert generated.feats.tolist() == out.feats.tolist()
    assert generated.stride == out.stride


def test_conv_rejects_ambiguous_contracts() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
    )

    with pytest.raises(ValueError, match='odd kernel_size'):
        subm_conv3d(x, mx.ones((2, 1, 1), dtype=mx.float32), kernel_size=2)

    with pytest.raises(ValueError, match='must divide'):
        conv_transpose3d(x, mx.ones((2, 1, 1), dtype=mx.float32))
