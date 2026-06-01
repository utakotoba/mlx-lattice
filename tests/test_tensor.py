import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import (  # noqa: E402
    SparseTensor,
    cat,
    prune,
    sparse_collate,
    topk_rows,
)


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


def test_sparse_tensor_reuses_explicit_coordinate_map():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.ones((2, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)
    y = SparseTensor(mx.array(coords.tolist(), dtype=mx.int32), feats + 1)

    reused = y.reuse_coords_from(x)

    assert reused.coord_key == x.coord_key
    assert reused.coord_manager is x.coord_manager
    assert reused.inverse_map(x).tolist() == [0, 1]


def test_sparse_tensor_queries_coordinate_rows():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.ones((3, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)
    queries = mx.array(
        [[0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]],
        dtype=mx.int32,
    )

    assert x.lookup_coords(queries).tolist() == [1, -1, 2]
    assert x.contains_coords(queries).tolist() == [True, False, True]


def test_sparse_tensor_replace_and_astype():
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.array([[1.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = x.replace(feats=feats + 1).astype(mx.float16)

    assert out.shape == (1, 1)
    assert out.dtype == mx.float16
    assert out.coords.tolist() == coords.tolist()


def test_sparse_tensor_add_and_cat_reuse_coordinates():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.ones((2, 1), dtype=mx.float32)
    x = SparseTensor(coords, feats)

    y = x.replace(feats=feats * 2)
    summed = x + y
    joined = cat([x, y])

    assert summed.feats.tolist() == [[3.0], [3.0]]
    assert joined.feats.tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert joined.coord_key == x.coord_key


def test_sparse_collate_decompose_topk_and_prune():
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[0.5], [2.0]], dtype=mx.float32),
            mx.array([[3.0], [1.0]], dtype=mx.float32),
        ],
    )

    rows = topk_rows(x, [1, 1])
    out = prune(x, rows)

    assert x.coords.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [1, 3, 0, 0],
    ]
    assert [part.tolist() for part in x.decomposed_coordinates] == [
        [[0, 0, 0], [1, 0, 0]],
        [[2, 0, 0], [3, 0, 0]],
    ]
    assert out.feats.tolist() == [[2.0], [3.0]]
