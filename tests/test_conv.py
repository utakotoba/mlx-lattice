from typing import cast

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import (  # noqa: E402
    SparseTensor,
    conv3d,
    generative_conv_transpose3d,
    pool3d,
)


def assert_allclose(actual, expected, *, rtol=1e-5, atol=1e-6):
    mx.eval(actual)
    assert mx.allclose(actual, expected, rtol=rtol, atol=atol)


def test_conv3d_k3s1_identity_center_weight():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.zeros((27, 1, 1), dtype=mx.float32)
    weight = mx.concatenate(
        [weight[:13], mx.ones((1, 1, 1), dtype=mx.float32), weight[14:]],
        axis=0,
    )

    out = conv3d(x, weight, kernel_size=3, stride=1)

    assert out.coords.tolist() == coords.tolist()
    assert_allclose(out.feats, feats)


def test_conv3d_k1s1_uses_pointwise_path():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    weight = mx.array([[[2.0], [3.0]]], dtype=mx.float32)
    bias = mx.array([1.0], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = conv3d(x, weight, bias, kernel_size=1, stride=1)

    assert out.coords.tolist() == coords.tolist()
    assert out.coord_key == x.coord_key
    assert len(x._maps) == 0
    assert_allclose(out.feats, mx.array([[9.0], [19.0]], dtype=mx.float32))


def test_conv3d_padding_shifts_sparse_window():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.ones((1, 1, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = conv3d(x, weight, kernel_size=1, stride=1, padding=(1, 0, 0))

    assert out.coords.tolist() == coords.tolist()
    assert_allclose(out.feats, mx.array([[0.0], [1.0], [2.0]]))


def test_conv3d_dilation_expands_sparse_window():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.ones((27, 1, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = conv3d(x, weight, kernel_size=3, stride=1, dilation=(2, 1, 1))

    assert out.coords.tolist() == coords.tolist()
    assert_allclose(out.feats, mx.array([[3.0], [6.0], [5.0]]))


def test_conv3d_k3s1_neighbor_sum():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)
    weight = mx.ones((27, 1, 1), dtype=mx.float32)

    out = conv3d(x, weight, kernel_size=3, stride=1)

    assert_allclose(out.feats, mx.array([[3.0], [6.0], [5.0]]))


def test_conv3d_backward_for_features_and_weight():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weight = mx.ones((27, 1, 1), dtype=mx.float32)

    def loss_feats(feats):
        x = SparseTensor(coords, feats)
        return mx.sum(conv3d(x, weight, kernel_size=3).feats)

    def loss_weight(weight):
        x = SparseTensor(coords, feats)
        return mx.sum(conv3d(x, weight, kernel_size=3).feats)

    assert_allclose(
        mx.grad(loss_feats)(feats),
        mx.array([[2.0], [3.0], [2.0]], dtype=mx.float32),
    )
    assert_allclose(mx.sum(mx.grad(loss_weight)(weight)), mx.array(14.0))


def test_pool3d_k2s2():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = pool3d(x, kernel_size=2, stride=2)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert_allclose(out.feats, mx.array([[3.0], [3.0]]))


def test_pool3d_center_kernel_does_not_mix_channels():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=mx.float32,
    )
    x = SparseTensor(coords, feats)

    out = pool3d(x, kernel_size=3, stride=1)

    assert out.coords.tolist() == coords.tolist()
    assert_allclose(
        out.feats,
        mx.array(
            [[3.0, 30.0], [6.0, 60.0], [5.0, 50.0]],
            dtype=mx.float32,
        ),
    )


def test_generative_conv_transpose3d_k2s2():
    coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[2.0]], dtype=mx.float32)
    weight = mx.ones((8, 1, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats, stride=2)

    out = generative_conv_transpose3d(x, weight, kernel_size=2, stride=2)
    coords_out = cast(list[list[int]], out.coords.tolist())

    assert out.stride == (1, 1, 1)
    assert out.coords.shape == (8, 4)
    assert coords_out[0] == [0, 2, 0, 0]
    assert coords_out[-1] == [0, 3, 1, 1]
    assert_allclose(out.feats, mx.full((8, 1), 2.0, dtype=mx.float32))


def test_generative_conv_transpose3d_backward():
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.array([[2.0]], dtype=mx.float32)
    weight = mx.ones((8, 1, 1), dtype=mx.float32)

    def loss_feats(feats):
        x = SparseTensor(coords, feats, stride=2)
        return mx.sum(generative_conv_transpose3d(x, weight).feats)

    def loss_weight(weight):
        x = SparseTensor(coords, feats, stride=2)
        return mx.sum(generative_conv_transpose3d(x, weight).feats)

    assert_allclose(mx.grad(loss_feats)(feats), mx.array([[8.0]]))
    assert_allclose(
        mx.grad(loss_weight)(weight).reshape((-1,)),
        mx.full((8,), 2.0, dtype=mx.float32),
    )
