from __future__ import annotations

import pytest

from mlx_lattice.core import (
    KernelRelation,
    KernelSpec,
    NeighborEdges,
    NeighborRelation,
    RelationEdges,
)
from tests.support import mx


def test_kernel_spec_normalizes_and_classifies_common_paths() -> None:
    pointwise = KernelSpec(size=1)
    subm = KernelSpec(size=(3, 3, 3))

    assert pointwise.size == (1, 1, 1)
    assert pointwise.volume == 1
    assert pointwise.is_pointwise
    assert pointwise.is_centered_submanifold
    assert subm.volume == 27
    assert subm.is_centered_submanifold
    assert not KernelSpec(size=2).is_centered_submanifold


def test_kernel_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match='kernel_size'):
        KernelSpec(size=0)
    with pytest.raises(ValueError, match='stride'):
        KernelSpec(stride=0)
    with pytest.raises(ValueError, match='padding'):
        KernelSpec(padding=-1)
    with pytest.raises(ValueError, match='dilation'):
        KernelSpec(dilation=0)


def test_kernel_relation_accepts_and_validates_edge_contract() -> None:
    rows = mx.array([0, 1], dtype=mx.int32)
    short = mx.array([0], dtype=mx.int32)
    out_coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )

    relation = KernelRelation(
        rows,
        rows,
        rows,
        kernel_offsets=((0, 0, 0), (1, 0, 0)),
        out_coords=out_coords,
        n_in_capacity=2,
    )

    assert relation.edge_capacity == 2
    assert isinstance(relation.edges, RelationEdges)
    assert relation.n_out_capacity == 2
    assert relation.n_in_capacity == 2
    assert relation.n_kernels == 2

    with pytest.raises(ValueError, match='same row count'):
        KernelRelation(rows, short, rows)
    with pytest.raises(ValueError, match='n_kernels'):
        KernelRelation(
            rows, rows, rows, kernel_offsets=((0, 0, 0),), n_kernels=2
        )


def test_neighbor_relation_accepts_and_validates_query_contract() -> None:
    rows = mx.array([0, 1], dtype=mx.int32)
    short = mx.array([0], dtype=mx.int32)
    distances = mx.array([1.0, 4.0], dtype=mx.float32)

    relation = NeighborRelation(
        rows,
        rows,
        rows,
        distances,
        n_query_capacity=2,
        n_source_capacity=3,
        max_neighbors=2,
    )

    assert relation.edge_capacity == 2
    assert isinstance(relation.edges, NeighborEdges)
    assert relation.n_query_capacity == 2
    assert relation.n_source_capacity == 3
    assert relation.max_neighbors == 2
    assert relation.edge_count.tolist() == [2]
    assert relation.query_count.tolist() == [0]

    with pytest.raises(ValueError, match='same row count'):
        NeighborRelation(rows, short, rows, distances)
    with pytest.raises(ValueError, match='distances'):
        NeighborRelation(rows, rows, rows, rows)
