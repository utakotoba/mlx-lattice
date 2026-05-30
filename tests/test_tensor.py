import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import SparseTensor  # noqa: E402


def test_sparse_tensor_validates_shape():
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.ones((1, 2), dtype=mx.float32)

    x = SparseTensor(coords, feats, stride=(1, 2, 3))

    assert x.coords is coords
    assert x.feats is feats
    assert x.stride == (1, 2, 3)
    assert x.n_points == 1
    assert x.channels == 2
    assert x.shape == (1, 2)
    assert x.dtype == mx.float32


def test_sparse_tensor_reuses_kernel_map():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.ones((2, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)

    first = x.kernel_map(kernel_size=3, stride=1)
    second = x.kernel_map(kernel_size=3, stride=1)

    assert first is second


def test_sparse_tensor_replace_and_astype():
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = x.replace(feats=feats + 1).astype(mx.float16)

    assert out.shape == (1, 1)
    assert out.dtype == mx.float16
    assert out.coords.tolist() == coords.tolist()
