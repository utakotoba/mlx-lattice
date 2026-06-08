from __future__ import annotations

# ruff: noqa: E402, I001

import pytest

mx = pytest.importorskip('mlx.core')

import mlx_lattice
from mlx_lattice import SparseTensor
from mlx_lattice import nn as lnn
from mlx_lattice.ops import sparse_collate


def _tensor() -> SparseTensor:
    return SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
    )


def test_feature_modules_preserve_sparse_identity_and_own_parameters() -> (
    None
):
    x = _tensor()
    layer = lnn.Linear(2, 2)
    layer.weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    layer.bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = lnn.ReLU()(layer(x))

    assert out.feats.tolist() == [[9.0, 18.0], [19.0, 42.0]]
    assert out.coord_key == x.coord_key
    assert out.coord_manager is x.coord_manager
    assert 'weight' in layer
    assert 'bias' in layer


def test_normalization_and_dropout_modules_delegate_feature_state() -> None:
    x = _tensor()
    norm = lnn.BatchNorm(2, affine=True, track_running_stats=False)
    norm.weight = mx.array([2.0, 3.0], dtype=mx.float32)
    norm.bias = mx.array([0.5, -0.5], dtype=mx.float32)

    normalized = norm(x)
    dropped = lnn.Dropout(p=0.5)(x)

    assert normalized.coord_key == x.coord_key
    assert normalized.feats.shape == x.feats.shape
    assert dropped.coord_key == x.coord_key
    assert dropped.feats.shape == x.feats.shape
    assert lnn.LayerNorm(2)(x).shape == x.shape
    assert lnn.RMSNorm(2)(x).shape == x.shape


def test_convolution_modules_wrap_public_sparse_ops() -> None:
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )
    conv = lnn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    conv.weight = mx.ones((3, 1, 1), dtype=mx.float32)
    subm = lnn.SubmConv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    subm.weight = conv.weight

    out = conv(x)
    subm_out = subm(x)

    assert out.feats.tolist() == [[3.0], [6.0], [5.0]]
    assert out.coord_manager is x.coord_manager
    assert subm_out.coord_key == x.coord_key
    assert subm_out.feats.tolist() == out.feats.tolist()


def test_transpose_and_pool_modules_wrap_sparse_policies() -> None:
    x = SparseTensor(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[4.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    transposed = lnn.ConvTranspose3d(
        1,
        1,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
        bias=False,
    )
    transposed.weight = mx.array([[[2.0]], [[3.0]]], dtype=mx.float32)

    out = transposed(x)
    pooled = lnn.SumPool3d(kernel_size=1, stride=1)(out)

    assert out.coords.tolist() == [[0, 2, 0, 0], [0, 3, 0, 0]]
    assert out.feats.tolist() == [[8.0], [12.0]]
    assert pooled.feats.tolist() == out.feats.tolist()


def test_global_pool_modules_return_dense_batch_features() -> None:
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[1.0], [2.0]], dtype=mx.float32),
            mx.array([[3.0], [5.0]], dtype=mx.float32),
        ],
    )

    assert lnn.GlobalSumPool()(x).tolist() == [[3.0], [8.0]]
    assert lnn.GlobalAvgPool()(x).tolist() == [[1.5], [4.0]]
    assert lnn.GlobalMaxPool()(x).tolist() == [[2.0], [5.0]]
    assert mlx_lattice.nn is lnn
