from __future__ import annotations

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    build_kernel_relation,
    build_knn_relation,
    conv3d,
    sum_pool3d,
)
from tests.support import BackendRun, mx

pytestmark = [pytest.mark.backend]


def test_sparse_convolution_rejects_unsupported_coord_dtype(
    backend: BackendRun,
) -> None:
    if backend.backend.name == 'cpu':
        pytest.skip('CPU accepts int64 coordinates')

    def run() -> None:
        x = SparseTensor(
            mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int64),
            mx.array([[1.0], [2.0]], dtype=mx.float32),
        )
        weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)
        with pytest.raises(ValueError, match='sparse convolution'):
            conv3d(x, weight, kernel_size=(3, 1, 1))

    backend(run)


def test_sparse_pooling_rejects_unsupported_coord_dtype(
    backend: BackendRun,
) -> None:
    if backend.backend.name == 'cpu':
        pytest.skip('CPU accepts int64 coordinates')

    def run() -> None:
        x = SparseTensor(
            mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int64),
            mx.array([[1.0], [2.0]], dtype=mx.float32),
        )
        with pytest.raises(ValueError, match='sparse pooling'):
            sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)

    backend(run)


def test_coordinate_kernels_reject_unsupported_coord_dtype(
    backend: BackendRun,
) -> None:
    if backend.backend.name == 'cpu':
        pytest.skip('CPU accepts int64 coordinates')

    def run() -> None:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int64,
        )
        relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
        assert relation.out_coords is not None
        with pytest.raises(ValueError, match='coordinate kernels'):
            mx.eval(relation.out_coords, relation.counts)

    backend(run)


def test_neighbor_kernels_reject_unsupported_coord_dtype(
    backend: BackendRun,
) -> None:
    if backend.backend.name == 'cpu':
        pytest.skip('CPU accepts int64 coordinates')

    def run() -> None:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int64,
        )
        relation = build_knn_relation(coords, k=1)
        with pytest.raises(ValueError, match='coordinate kernels'):
            mx.eval(relation.edges.query_rows, relation.counts)

    backend(run)
