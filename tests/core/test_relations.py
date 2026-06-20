from __future__ import annotations

import pytest

from mlx_lattice.core import (
    KernelRelation,
    KernelSpec,
    NeighborEdges,
    NeighborRelation,
    RelationCSRView,
    RelationEdges,
    SparseRelationContract,
)
from mlx_lattice.core.coords.builders import (
    build_kernel_relation,
    build_target_kernel_relation,
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
    assert isinstance(relation.contract, SparseRelationContract)
    assert isinstance(relation.edges, RelationEdges)
    assert relation.n_out_capacity == 2
    assert relation.n_in_capacity == 2
    assert relation.n_kernels == 2
    assert isinstance(relation.output_csr, RelationCSRView)
    assert isinstance(relation.input_csr, RelationCSRView)
    assert isinstance(relation.kernel_csr, RelationCSRView)
    assert relation.out_view is relation.output_csr
    assert relation.in_view is relation.input_csr
    assert relation.kernel_view is relation.kernel_csr
    assert relation.row_offsets is relation.output_csr.row_offsets

    with pytest.raises(ValueError, match='same row count'):
        KernelRelation(rows, short, rows)
    with pytest.raises(ValueError, match='n_kernels'):
        KernelRelation(
            rows, rows, rows, kernel_offsets=((0, 0, 0),), n_kernels=2
        )


def test_kernel_relation_materializes_implicit_gemm_view_lazily() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))

    assert relation.implicit_gemm is None

    view = relation.require_implicit_gemm()
    mx.eval(view.out_in_map, view.row_masks)

    assert view is relation.require_implicit_gemm()
    assert view.out_in_map.tolist() == [
        [-1, 0, 1],
        [0, 1, 2],
        [1, 2, -1],
    ]
    assert view.row_masks.tolist() == [[6], [7], [3]]


def test_target_relation_materializes_implicit_gemm_from_target_coords() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    relation = build_target_kernel_relation(
        coords,
        target,
        kernel_size=(3, 1, 1),
    )

    view = relation.require_implicit_gemm()
    mx.eval(view.out_in_map, view.row_masks)

    assert view.out_in_map.tolist() == [
        [0, 1, 2],
        [2, -1, -1],
    ]
    assert view.row_masks.tolist() == [[7], [1]]


def test_implicit_gemm_row_masks_scale_past_single_word() -> None:
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    relation = build_kernel_relation(coords, kernel_size=(33, 1, 1))

    view = relation.require_implicit_gemm()
    mx.eval(view.out_in_map, view.row_masks)

    assert view.out_in_map.shape == (1, 33)
    assert view.row_masks.shape == (1, 2)
    assert view.out_in_map.tolist()[0][16] == 0
    assert view.row_masks.tolist() == [[1 << 16, 0]]


def test_sorted_implicit_gemm_view_has_independent_cache() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))

    implicit = relation.require_implicit_gemm()
    sorted_view = relation.require_sorted_implicit_gemm()
    mx.eval(
        implicit.out_in_map,
        implicit.row_masks,
        sorted_view.sorted_out_in_map,
        sorted_view.reorder_rows,
        sorted_view.tile_masks,
    )

    assert relation.require_implicit_gemm() is implicit
    assert relation.require_sorted_implicit_gemm() is sorted_view
    assert sorted_view.reorder_rows.tolist() == [2, 0, 1]
    assert sorted_view.sorted_out_in_map.tolist() == [
        [1, 2, -1],
        [-1, 0, 1],
        [0, 1, 2],
    ]
    assert sorted_view.tile_masks.tolist() == [7]


def test_neighbor_relation_accepts_and_validates_query_contract() -> None:
    rows = mx.array([0, 1], dtype=mx.int32)
    short = mx.array([0], dtype=mx.int32)
    distances = mx.array([1.0, 4.0], dtype=mx.float32)
    row_offsets = mx.array([0, 1, 2], dtype=mx.int32)

    relation = NeighborRelation(
        rows,
        rows,
        rows,
        distances,
        row_offsets=row_offsets,
        n_query_capacity=2,
        n_source_capacity=3,
        max_neighbors=2,
    )

    assert relation.edge_capacity == 2
    assert isinstance(relation.edges, NeighborEdges)
    assert relation.n_query_capacity == 2
    assert relation.n_source_capacity == 3
    assert relation.max_neighbors == 2
    assert relation.row_offsets is row_offsets
    assert relation.edge_count.tolist() == [2]
    assert relation.query_count.tolist() == [0]

    with pytest.raises(ValueError, match='same row count'):
        NeighborRelation(rows, short, rows, distances)
    with pytest.raises(ValueError, match='distances'):
        NeighborRelation(rows, rows, rows, rows)
