from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice.core.types import Triple


@dataclass(frozen=True, slots=True)
class RelationEdges:
    """Diagnostic edge arrays for a logical sparse neighborhood relation."""

    in_rows: mx.array
    out_rows: mx.array
    kernel_ids: mx.array

    def __post_init__(self) -> None:
        _validate_row_array(self.in_rows, name='in_rows')
        _validate_row_array(self.out_rows, name='out_rows')
        _validate_row_array(self.kernel_ids, name='kernel_ids')
        _require_same_rows(
            self.in_rows,
            self.out_rows,
            self.kernel_ids,
            names=('in_rows', 'out_rows', 'kernel_ids'),
        )

    @property
    def capacity(self) -> int:
        return int(self.in_rows.shape[0])


@dataclass(frozen=True, slots=True)
class NeighborEdges:
    """Semantic neighbor edge arrays for query/source relations."""

    query_rows: mx.array
    source_rows: mx.array
    neighbor_ids: mx.array

    def __post_init__(self) -> None:
        _validate_row_array(self.query_rows, name='query_rows')
        _validate_row_array(self.source_rows, name='source_rows')
        _validate_row_array(self.neighbor_ids, name='neighbor_ids')
        _require_same_rows(
            self.query_rows,
            self.source_rows,
            self.neighbor_ids,
            names=('query_rows', 'source_rows', 'neighbor_ids'),
        )

    @property
    def capacity(self) -> int:
        return int(self.query_rows.shape[0])


@dataclass(frozen=True, slots=True, init=False)
class KernelRelation:
    edges: RelationEdges
    counts: mx.array
    kernel_offsets: tuple[Triple, ...]
    out_coords: mx.array | None = None
    n_in_capacity: int | None = None
    n_out_capacity: int | None = None
    n_kernels: int | None = None

    def __init__(
        self,
        in_rows: mx.array,
        out_rows: mx.array,
        kernel_ids: mx.array,
        *,
        counts: mx.array | None = None,
        kernel_offsets: tuple[Triple, ...] = (),
        out_coords: mx.array | None = None,
        n_in_capacity: int | None = None,
        n_out_capacity: int | None = None,
        n_kernels: int | None = None,
    ) -> None:
        if out_coords is not None:
            _validate_coords(out_coords, name='out_coords')
        if counts is None:
            counts = mx.array(
                [
                    in_rows.shape[0],
                    0 if out_coords is None else out_coords.shape[0],
                ],
                dtype=mx.int32,
            )
        _validate_counts(counts)

        edges = RelationEdges(in_rows, out_rows, kernel_ids)
        normalized_kernel_offsets = tuple(
            (int(x), int(y), int(z)) for x, y, z in kernel_offsets
        )
        normalized_n_in_capacity = _optional_count(
            n_in_capacity, 'n_in_capacity'
        )
        normalized_n_out_capacity = _optional_count(
            n_out_capacity, 'n_out_capacity'
        )
        normalized_n_kernels = _optional_count(n_kernels, 'n_kernels')
        if (
            normalized_kernel_offsets
            and normalized_n_kernels is not None
            and len(normalized_kernel_offsets) != normalized_n_kernels
        ):
            raise ValueError('n_kernels must match kernel_offsets.')
        if normalized_kernel_offsets:
            normalized_n_kernels = len(normalized_kernel_offsets)
        if out_coords is not None:
            out_coord_capacity = int(out_coords.shape[0])
            if (
                normalized_n_out_capacity is not None
                and normalized_n_out_capacity != out_coord_capacity
            ):
                raise ValueError(
                    'n_out_capacity must match out_coords capacity.'
                )
            normalized_n_out_capacity = out_coord_capacity

        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'counts', counts)
        object.__setattr__(
            self, 'kernel_offsets', normalized_kernel_offsets
        )
        object.__setattr__(self, 'out_coords', out_coords)
        object.__setattr__(self, 'n_in_capacity', normalized_n_in_capacity)
        object.__setattr__(
            self, 'n_out_capacity', normalized_n_out_capacity
        )
        object.__setattr__(self, 'n_kernels', normalized_n_kernels)

    @property
    def edge_capacity(self) -> int:
        return self.edges.capacity

    @property
    def edge_count(self) -> mx.array:
        return self.counts[:1]

    @property
    def out_count(self) -> mx.array:
        return self.counts[1:2]


@dataclass(frozen=True, slots=True, init=False)
class NeighborRelation:
    edges: NeighborEdges
    distances: mx.array
    counts: mx.array
    n_query_capacity: int | None = None
    n_source_capacity: int | None = None
    max_neighbors: int | None = None

    def __init__(
        self,
        query_rows: mx.array,
        source_rows: mx.array,
        neighbor_ids: mx.array,
        distances: mx.array,
        *,
        counts: mx.array | None = None,
        n_query_capacity: int | None = None,
        n_source_capacity: int | None = None,
        max_neighbors: int | None = None,
    ) -> None:
        edges = NeighborEdges(query_rows, source_rows, neighbor_ids)
        _validate_distance_array(distances)
        _require_same_rows(
            query_rows,
            distances,
            names=('query_rows', 'distances'),
        )
        if counts is None:
            counts = mx.array([query_rows.shape[0], 0], dtype=mx.int32)
        _validate_counts(counts)

        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'distances', distances)
        object.__setattr__(self, 'counts', counts)
        object.__setattr__(
            self,
            'n_query_capacity',
            _optional_count(n_query_capacity, 'n_query_capacity'),
        )
        object.__setattr__(
            self,
            'n_source_capacity',
            _optional_count(n_source_capacity, 'n_source_capacity'),
        )
        object.__setattr__(
            self,
            'max_neighbors',
            _optional_count(max_neighbors, 'max_neighbors'),
        )

    @property
    def edge_capacity(self) -> int:
        return self.edges.capacity

    @property
    def edge_count(self) -> mx.array:
        return self.counts[:1]

    @property
    def query_count(self) -> mx.array:
        return self.counts[1:2]


# MARK: - helpers


def _validate_row_array(value: mx.array, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f'{name} must have shape (E,).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_coords(value: mx.array, *, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_counts(value: mx.array) -> None:
    if value.shape != (2,) or value.dtype != mx.int32:
        raise ValueError(
            'relation counts must have shape (2,) and int32 dtype.'
        )


def _validate_distance_array(value: mx.array) -> None:
    if value.ndim != 1:
        raise ValueError('distances must have shape (E,).')
    if value.dtype not in (mx.float32, mx.float64):
        raise ValueError('distances must be float32 or float64.')


def _require_same_rows(
    first: mx.array,
    *rest: mx.array,
    names: tuple[str, ...],
) -> None:
    rows = int(first.shape[0])
    for name, value in zip(names[1:], rest, strict=True):
        if int(value.shape[0]) != rows:
            raise ValueError(
                f'{names[0]} and {name} must have the same row count.'
            )


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f'{name} must be non-negative.')
    return normalized
