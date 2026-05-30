import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor, conv3d  # noqa: E402
from mlx_lattice import nn as lnn  # noqa: E402


def assert_allclose(actual, expected, *, rtol=1e-5, atol=1e-6):
    mx.eval(actual)
    assert mx.allclose(actual, expected, rtol=rtol, atol=atol)


def make_tensor():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    return SparseTensor(coords, feats)


def center_weight() -> mx.array:
    weight = mx.zeros((27, 1, 1), dtype=mx.float32)
    weight = mx.concatenate(
        [weight[:13], mx.ones((1, 1, 1), dtype=mx.float32), weight[14:]],
        axis=0,
    )
    return mx.moveaxis(weight.reshape((3, 3, 3, 1, 1)), -1, 0)


def test_conv3d_accepts_mlx_weight_layout():
    x = make_tensor()

    out = conv3d(x, center_weight(), kernel_size=3)

    assert_allclose(out.feats, x.feats)


def test_nn_conv3d_matches_mlx_shape_convention():
    x = make_tensor()
    layer = lnn.Conv3d(1, 1, kernel_size=3)
    layer.weight = center_weight()
    layer.bias = mx.array([0.5], dtype=mx.float32)

    out = layer(x)

    assert layer.weight.shape == (1, 3, 3, 3, 1)
    assert_allclose(out.feats, x.feats + 0.5)


def test_nn_sum_pool3d():
    x = make_tensor()
    pool = lnn.SumPool3d(kernel_size=2, stride=2)

    out = pool(x)

    assert out.coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert_allclose(out.feats, mx.array([[3.0], [3.0]], dtype=mx.float32))
