from __future__ import annotations

from typing import cast

import pytest

from mlx_lattice.core import CoordinateManager, SparseTensor
from mlx_lattice.ops import (
    build_generative_relation,
    build_kernel_relation,
    build_knn_relation,
    build_radius_relation,
    build_target_kernel_relation,
    build_transposed_kernel_relation,
    downsample_coords,
    intersection_coords,
    kernel_offsets,
    kernel_relation,
    knn_relation,
    lookup_coords,
    radius_relation,
    union_coords,
)
from tests.support import assert_nested_close, mx, run_with_gpu_default


def _active_rows(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0])].tolist())


def _active_row_offsets(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0]) + 1].tolist())


def _active_floats(values: mx.array, count: mx.array) -> list[float]:
    return cast('list[float]', values[: int(count.tolist()[0])].tolist())


def _active_coords(values: mx.array, count: mx.array) -> list[list[int]]:
    return cast(
        'list[list[int]]', values[: int(count.tolist()[0])].tolist()
    )


def _coord_set_rows(value: object) -> list[list[int]]:
    coords = value.coords
    count = value.active_rows
    return _active_coords(coords, count)


def test_coordinate_set_primitives_preserve_first_seen_order() -> None:
    lhs = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    rhs = mx.array(
        [[0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )

    downsampled = downsample_coords(rhs, stride=2)
    assert downsampled.capacity == 2
    assert downsampled.active_count is downsampled.active_rows
    assert _coord_set_rows(downsampled) == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    assert _coord_set_rows(union_coords(lhs, rhs)) == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert _coord_set_rows(intersection_coords(lhs, rhs)) == [[0, 1, 0, 0]]
    assert lookup_coords(lhs, rhs).tolist() == [1, -1]


def test_kernel_offsets_and_relation_builders_emit_expected_edges() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))

    assert kernel_offsets((3, 1, 1)) == ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
    assert relation.out_coords is not None
    assert relation.counts.tolist() == [7, 3]
    assert (
        _active_coords(relation.out_coords, relation.out_count)
        == coords.tolist()
    )
    assert _active_rows(relation.edges.in_rows, relation.edge_count) == [
        0,
        1,
        0,
        1,
        2,
        1,
        2,
    ]
    assert _active_rows(relation.edges.out_rows, relation.edge_count) == [
        0,
        0,
        1,
        1,
        1,
        2,
        2,
    ]
    assert _active_rows(relation.edges.kernel_ids, relation.edge_count) == [
        1,
        2,
        0,
        1,
        2,
        0,
        1,
    ]
    assert _active_row_offsets(
        relation.row_offsets, relation.out_count
    ) == [
        0,
        2,
        5,
        7,
    ]
    assert relation.in_row_offsets.tolist() == [0, 2, 5, 7]
    assert _active_rows(relation.in_edge_ids, relation.edge_count) == [
        0,
        2,
        1,
        3,
        5,
        4,
        6,
    ]
    assert relation.kernel_row_offsets.tolist() == [0, 2, 5, 7]
    assert _active_rows(relation.kernel_edge_ids, relation.edge_count) == [
        2,
        5,
        0,
        3,
        6,
        1,
        4,
    ]


def test_target_kernel_relation_uses_explicit_output_coordinates() -> None:
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

    assert relation.out_coords is not None
    assert _active_coords(relation.out_coords, relation.out_count) == [
        [0, 1, 0, 0],
        [0, 3, 0, 0],
    ]
    assert relation.counts.tolist() == [4, 2]
    assert _active_rows(relation.edges.in_rows, relation.edge_count) == [
        0,
        1,
        2,
        2,
    ]
    assert _active_rows(relation.edges.out_rows, relation.edge_count) == [
        0,
        0,
        0,
        1,
    ]
    assert _active_rows(relation.edges.kernel_ids, relation.edge_count) == [
        0,
        1,
        2,
        0,
    ]
    assert _active_row_offsets(
        relation.row_offsets, relation.out_count
    ) == [0, 3, 4]


def test_strided_and_transposed_relations_define_output_policy() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    strided = build_kernel_relation(coords, kernel_size=1, stride=2)
    generated = build_generative_relation(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    transposed = build_transposed_kernel_relation(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )

    assert strided.out_coords is not None
    assert strided.counts.tolist() == [2, 2]
    assert _active_coords(strided.out_coords, strided.out_count) == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    assert _active_rows(strided.edges.in_rows, strided.edge_count) == [
        0,
        2,
    ]
    assert generated.out_coords is not None
    assert generated.counts.tolist() == [2, 2]
    assert _active_coords(generated.out_coords, generated.out_count) == [
        [0, 2, 0, 0],
        [0, 3, 0, 0],
    ]
    assert transposed.out_coords is not None
    assert _active_coords(transposed.out_coords, transposed.out_count) == (
        _active_coords(generated.out_coords, generated.out_count)
    )


def test_knn_and_radius_relations_define_neighbor_query_contract() -> None:
    source = mx.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 5, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=mx.int32,
    )
    query = mx.array(
        [[0, 1, 0, 0], [0, 4, 0, 0], [1, 1, 0, 0]],
        dtype=mx.int32,
    )

    knn = build_knn_relation(source, query, k=2)
    radius = build_radius_relation(source, query, radius=1.5)
    capped = build_radius_relation(
        source,
        query,
        radius=1.5,
        max_neighbors=1,
    )

    assert knn.counts.tolist() == [5, 3]
    assert _active_rows(knn.edges.query_rows, knn.edge_count) == [
        0,
        0,
        1,
        1,
        2,
    ]
    assert _active_rows(knn.edges.source_rows, knn.edge_count) == [
        0,
        1,
        2,
        1,
        3,
    ]
    assert _active_rows(knn.edges.neighbor_ids, knn.edge_count) == [
        0,
        1,
        0,
        1,
        0,
    ]
    assert _active_floats(knn.distances, knn.edge_count) == [
        1.0,
        1.0,
        1.0,
        4.0,
        1.0,
    ]
    assert _active_row_offsets(knn.row_offsets, knn.query_count) == [
        0,
        2,
        4,
        5,
    ]
    assert knn.n_query_capacity == 3
    assert knn.n_source_capacity == 4
    assert knn.max_neighbors == 2

    assert radius.counts.tolist() == [4, 3]
    assert _active_rows(radius.edges.query_rows, radius.edge_count) == [
        0,
        0,
        1,
        2,
    ]
    assert _active_rows(radius.edges.source_rows, radius.edge_count) == [
        0,
        1,
        2,
        3,
    ]
    assert _active_rows(radius.edges.neighbor_ids, radius.edge_count) == [
        0,
        1,
        0,
        0,
    ]
    assert _active_floats(radius.distances, radius.edge_count) == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert _active_row_offsets(radius.row_offsets, radius.query_count) == [
        0,
        2,
        3,
        4,
    ]
    assert capped.counts.tolist() == [3, 3]
    assert _active_rows(capped.edges.source_rows, capped.edge_count) == [
        0,
        2,
        3,
    ]


def test_neighbor_relations_respect_active_rows_and_tensor_wrappers() -> (
    None
):
    source = mx.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 99, 0, 0]],
        dtype=mx.int32,
    )
    query = mx.array(
        [[0, 1, 0, 0], [0, 50, 0, 0]],
        dtype=mx.int32,
    )
    source_active = mx.array([2], dtype=mx.int32)
    query_active = mx.array([1], dtype=mx.int32)

    relation = build_knn_relation(
        source,
        query,
        source_active_rows=source_active,
        query_active_rows=query_active,
        k=3,
    )
    source_tensor = SparseTensor(
        source,
        mx.ones((3, 1)),
        active_rows=source_active,
    )
    query_tensor = SparseTensor(
        query,
        mx.ones((2, 1)),
        active_rows=query_active,
    )
    wrapped = knn_relation(source_tensor, query_tensor, k=3)
    radius = radius_relation(source_tensor, query_tensor, radius=1.0)

    assert relation.counts.tolist() == [2, 1]
    assert _active_rows(
        relation.edges.source_rows, relation.edge_count
    ) == [
        0,
        1,
    ]
    assert wrapped.counts.tolist() == relation.counts.tolist()
    assert _active_rows(wrapped.edges.source_rows, wrapped.edge_count) == [
        0,
        1,
    ]
    assert radius.counts.tolist() == [2, 1]

    mismatched = SparseTensor(query, mx.ones((2, 1)), stride=(2, 1, 1))
    with pytest.raises(ValueError, match='same coordinate stride'):
        knn_relation(source_tensor, mismatched, k=1)


def test_coordinate_manager_caches_tensor_kernel_relations() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    manager = CoordinateManager()
    key = manager.insert_coords(coords)

    first = manager.kernel_relation(key, kernel_size=(3, 1, 1))
    second = manager.kernel_relation(key, kernel_size=(3, 1, 1))
    tensor_relation = kernel_relation(
        SparseTensor(coords, mx.ones((2, 1))), kernel_size=(3, 1, 1)
    )

    assert first is second
    assert tensor_relation.kernel_offsets == first.kernel_offsets


def test_coordinate_manager_caches_target_kernel_relations() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    target = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    manager = CoordinateManager()
    key = manager.insert_coords(coords)
    target_key = manager.insert_coords(target)

    first = manager.target_kernel_relation(
        key, target_key, kernel_size=(3, 1, 1)
    )
    second = manager.target_kernel_relation(
        key, target_key, kernel_size=(3, 1, 1)
    )

    assert first is second
    assert first.counts.tolist() == [2, 1]


def test_metal_coordinate_primitives_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[int]],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ]:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
        mx.eval(
            relation.out_coords,
            relation.edges.in_rows,
            relation.edges.out_rows,
            relation.edges.kernel_ids,
            relation.row_offsets,
            relation.counts,
        )
        assert relation.out_coords is not None
        return (
            _active_coords(relation.out_coords, relation.out_count),
            _active_rows(relation.edges.in_rows, relation.edge_count),
            _active_rows(relation.edges.out_rows, relation.edge_count),
            _active_rows(relation.edges.kernel_ids, relation.edge_count),
            _active_row_offsets(relation.row_offsets, relation.out_count),
            cast('list[int]', relation.counts.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        [0, 1, 0, 1, 2, 1, 2],
        [0, 0, 1, 1, 1, 2, 2],
        [1, 2, 0, 1, 2, 0, 1],
        [0, 2, 5, 7],
        [7, 3],
    )


def test_metal_target_relation_matches_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[int]],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ]:
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
        mx.eval(
            relation.out_coords,
            relation.edges.in_rows,
            relation.edges.out_rows,
            relation.edges.kernel_ids,
            relation.row_offsets,
            relation.counts,
        )
        assert relation.out_coords is not None
        return (
            _active_coords(relation.out_coords, relation.out_count),
            _active_rows(relation.edges.in_rows, relation.edge_count),
            _active_rows(relation.edges.out_rows, relation.edge_count),
            _active_rows(relation.edges.kernel_ids, relation.edge_count),
            _active_row_offsets(relation.row_offsets, relation.out_count),
            cast('list[int]', relation.counts.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 1, 0, 0], [0, 3, 0, 0]],
        [0, 1, 2, 2],
        [0, 0, 0, 1],
        [0, 1, 2, 0],
        [0, 3, 4],
        [4, 2],
    )


def test_metal_coordinate_sets_preserve_stable_duplicate_semantics() -> (
    None
):
    def run() -> tuple[
        list[list[int]],
        list[list[int]],
        list[list[int]],
        list[int],
        list[list[int]],
    ]:
        lhs = mx.array(
            [
                [0, 2, 0, 0],
                [0, -1, 0, 0],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=mx.int32,
        )
        rhs = mx.array(
            [
                [0, -1, 0, 0],
                [0, 3, 0, 0],
                [0, 3, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=mx.int32,
        )
        queries = mx.array(
            [[0, 2, 0, 0], [0, -1, 0, 0], [0, 9, 0, 0]],
            dtype=mx.int32,
        )
        empty = mx.array([], dtype=mx.int32).reshape((0, 4))
        downsampled = downsample_coords(lhs, stride=(2, 1, 1))
        union = union_coords(lhs, rhs)
        intersection = intersection_coords(lhs, rhs)
        lookup = lookup_coords(lhs, queries)
        empty_union = union_coords(empty, empty)
        mx.eval(
            downsampled.coords,
            downsampled.active_rows,
            union.coords,
            union.active_rows,
            intersection.coords,
            intersection.active_rows,
            lookup,
            empty_union.coords,
            empty_union.active_rows,
        )
        return (
            _coord_set_rows(downsampled),
            _coord_set_rows(union),
            _coord_set_rows(intersection),
            cast('list[int]', lookup.tolist()),
            _coord_set_rows(empty_union),
        )

    assert run_with_gpu_default(run) == (
        [[0, 1, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
        [
            [0, 2, 0, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 3, 0, 0],
            [1, 1, 0, 0],
        ],
        [[0, -1, 0, 0], [1, 0, 0, 0]],
        [0, 1, -1],
        [],
    )


def test_metal_strided_relation_preserves_stable_edge_contract() -> None:
    def run() -> tuple[
        list[list[int]],
        list[int],
        list[int],
        list[int],
        list[int],
        list[int],
    ]:
        coords = mx.array(
            [[0, row, 0, 0] for row in range(6)],
            dtype=mx.int32,
        )
        relation = build_kernel_relation(
            coords,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
        )
        assert relation.out_coords is not None
        mx.eval(
            relation.out_coords,
            relation.edges.in_rows,
            relation.edges.out_rows,
            relation.edges.kernel_ids,
            relation.row_offsets,
            relation.counts,
        )
        return (
            _active_coords(relation.out_coords, relation.out_count),
            _active_rows(relation.edges.in_rows, relation.edge_count),
            _active_rows(relation.edges.out_rows, relation.edge_count),
            _active_rows(relation.edges.kernel_ids, relation.edge_count),
            _active_row_offsets(relation.row_offsets, relation.out_count),
            cast('list[int]', relation.counts.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        [0, 1, 1, 2, 3, 3, 4, 5],
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [0, 2, 5, 8],
        [8, 3],
    )


def test_metal_neighbor_relations_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[int],
        list[int],
        list[int],
        list[float],
        list[int],
        list[int],
    ]:
        source = mx.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 5, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=mx.int32,
        )
        query = mx.array(
            [[0, 1, 0, 0], [0, 4, 0, 0], [1, 1, 0, 0]],
            dtype=mx.int32,
        )
        knn = build_knn_relation(source, query, k=2)
        radius = build_radius_relation(source, query, radius=1.5)
        mx.eval(
            knn.edges.query_rows,
            knn.edges.source_rows,
            knn.edges.neighbor_ids,
            knn.distances,
            knn.row_offsets,
            knn.counts,
            radius.counts,
        )
        return (
            _active_rows(knn.edges.query_rows, knn.edge_count),
            _active_rows(knn.edges.source_rows, knn.edge_count),
            _active_rows(knn.edges.neighbor_ids, knn.edge_count),
            _active_floats(knn.distances, knn.edge_count),
            cast('list[int]', knn.counts.tolist()),
            cast('list[int]', radius.counts.tolist()),
        )

    (
        query_rows,
        source_rows,
        neighbor_ids,
        distances,
        knn_counts,
        radius_counts,
    ) = run_with_gpu_default(run)
    assert query_rows == [0, 0, 1, 1, 2]
    assert source_rows == [0, 1, 2, 1, 3]
    assert neighbor_ids == [0, 1, 0, 1, 0]
    assert_nested_close(distances, [1.0, 1.0, 1.0, 4.0, 1.0])
    assert knn_counts == [5, 3]
    assert radius_counts == [4, 3]


def test_metal_coordinate_primitives_reject_unsupported_coord_dtype() -> (
    None
):
    def run() -> None:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int64,
        )
        relation = build_kernel_relation(coords, kernel_size=(3, 1, 1))
        assert relation.out_coords is not None
        with pytest.raises(ValueError, match='Metal coordinate kernels'):
            mx.eval(relation.out_coords, relation.counts)

    run_with_gpu_default(run)


def test_metal_neighbor_relations_reject_unsupported_coord_dtype() -> None:
    def run() -> None:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int64,
        )
        relation = build_knn_relation(coords, k=1)
        with pytest.raises(ValueError, match='Metal coordinate kernels'):
            mx.eval(relation.edges.query_rows, relation.counts)

    run_with_gpu_default(run)
