from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_lattice.core.types import Triple

RelationKind = Literal['forward', 'target', 'transposed', 'generative']


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
class RelationCSRView:
    """CSR execution view over relation edge arrays."""

    row_offsets: mx.array
    edge_ids: mx.array | None = None

    def __post_init__(self) -> None:
        _validate_row_offsets(self.row_offsets)
        if self.edge_ids is not None:
            _validate_row_array(self.edge_ids, name='edge_ids')


RelationView = RelationCSRView


@dataclass(frozen=True, slots=True)
class RelationImplicitGemmView:
    """Reserved relation view for future implicit-GEMM/TensorOps kernels."""

    out_in_map: mx.array
    row_masks: mx.array

    def __post_init__(self) -> None:
        if self.out_in_map.ndim != 2:
            raise ValueError('out_in_map must have shape (Nout, K).')
        if self.out_in_map.dtype not in (mx.int32, mx.int64):
            raise ValueError('out_in_map must be int32 or int64.')
        if self.row_masks.ndim != 2:
            raise ValueError('row_masks must have shape (Nout, M).')
        if self.row_masks.dtype not in (mx.int32, mx.int64):
            raise ValueError('row_masks must be int32 or int64.')
        if int(self.row_masks.shape[0]) != int(self.out_in_map.shape[0]):
            raise ValueError(
                'row_masks must have one row per out_in_map row.'
            )
        expected_mask_words = (int(self.out_in_map.shape[1]) + 31) // 32
        if int(self.row_masks.shape[1]) != expected_mask_words:
            raise ValueError(
                'row_masks must have ceil(K / 32) words per output row.'
            )


@dataclass(frozen=True, slots=True)
class RelationSortedImplicitGemmView:
    """Tile-sorted implicit-GEMM execution view for TensorOps kernels."""

    sorted_out_in_map: mx.array
    sorted_kv_out_in_map: mx.array
    reorder_rows: mx.array
    tile_masks: mx.array

    def __post_init__(self) -> None:
        if self.sorted_out_in_map.ndim != 2:
            raise ValueError('sorted_out_in_map must have shape (Nout, K).')
        if self.sorted_out_in_map.dtype not in (mx.int32, mx.int64):
            raise ValueError('sorted_out_in_map must be int32 or int64.')
        if self.sorted_kv_out_in_map.ndim != 2:
            raise ValueError(
                'sorted_kv_out_in_map must have shape (K, Nout).'
            )
        if self.sorted_kv_out_in_map.dtype not in (mx.int32, mx.int64):
            raise ValueError('sorted_kv_out_in_map must be int32 or int64.')
        if int(self.sorted_kv_out_in_map.shape[0]) != int(
            self.sorted_out_in_map.shape[1]
        ) or int(self.sorted_kv_out_in_map.shape[1]) != int(
            self.sorted_out_in_map.shape[0]
        ):
            raise ValueError(
                'sorted_kv_out_in_map must be the K-major view of '
                'sorted_out_in_map.'
            )
        _validate_row_array(self.reorder_rows, name='reorder_rows')
        _validate_row_array(self.tile_masks, name='tile_masks')
        if int(self.reorder_rows.shape[0]) != int(
            self.sorted_out_in_map.shape[0]
        ):
            raise ValueError(
                'reorder_rows must have one row per sorted_out_in_map row.'
            )


@dataclass(frozen=True, slots=True)
class SparseRelationContract:
    """Logical sparse relation contract shared by all execution views."""

    counts: mx.array
    kernel_offsets: tuple[Triple, ...]
    out_coords: mx.array | None = None
    n_in_capacity: int | None = None
    n_out_capacity: int | None = None
    n_kernels: int | None = None
    source_coords: mx.array | None = None
    source_active_rows: mx.array | None = None
    target_coords: mx.array | None = None
    target_active_rows: mx.array | None = None
    stride: Triple = (1, 1, 1)
    padding: Triple = (0, 0, 0)
    kind: RelationKind = 'forward'

    def __post_init__(self) -> None:
        _validate_counts(self.counts)
        if self.out_coords is not None:
            _validate_coords(self.out_coords, name='out_coords')
        if self.source_coords is not None:
            _validate_coords(self.source_coords, name='source_coords')
        if self.target_coords is not None:
            _validate_coords(self.target_coords, name='target_coords')
        if self.source_active_rows is not None:
            _validate_active_rows(
                self.source_active_rows, name='source_active_rows'
            )
        if self.target_active_rows is not None:
            _validate_active_rows(
                self.target_active_rows, name='target_active_rows'
            )
        if self.kind not in (
            'forward',
            'target',
            'transposed',
            'generative',
        ):
            raise ValueError(
                "relation kind must be 'forward', 'target', "
                "'transposed', or 'generative'."
            )
        normalized_offsets = tuple(
            (int(x), int(y), int(z)) for x, y, z in self.kernel_offsets
        )
        object.__setattr__(self, 'kernel_offsets', normalized_offsets)
        object.__setattr__(
            self, 'stride', tuple(int(value) for value in self.stride)
        )
        object.__setattr__(
            self, 'padding', tuple(int(value) for value in self.padding)
        )
        n_in = _optional_count(self.n_in_capacity, 'n_in_capacity')
        n_out = _optional_count(self.n_out_capacity, 'n_out_capacity')
        n_kernels = _optional_count(self.n_kernels, 'n_kernels')
        if normalized_offsets:
            if n_kernels is not None and n_kernels != len(
                normalized_offsets
            ):
                raise ValueError('n_kernels must match kernel_offsets.')
            n_kernels = len(normalized_offsets)
        if self.out_coords is not None:
            out_capacity = int(self.out_coords.shape[0])
            if n_out is not None and n_out != out_capacity:
                raise ValueError(
                    'n_out_capacity must match out_coords capacity.'
                )
            n_out = out_capacity
        object.__setattr__(self, 'n_in_capacity', n_in)
        object.__setattr__(self, 'n_out_capacity', n_out)
        object.__setattr__(self, 'n_kernels', n_kernels)

    @property
    def edge_count(self) -> mx.array:
        return self.counts[:1]

    @property
    def out_count(self) -> mx.array:
        return self.counts[1:2]


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
    contract: SparseRelationContract
    edges: RelationEdges
    output_csr: RelationCSRView
    input_csr: RelationCSRView
    kernel_csr: RelationCSRView
    implicit_gemm: RelationImplicitGemmView | None = None
    sorted_implicit_gemm: RelationSortedImplicitGemmView | None = None

    def __init__(
        self,
        in_rows: mx.array,
        out_rows: mx.array,
        kernel_ids: mx.array,
        *,
        row_offsets: mx.array | None = None,
        counts: mx.array | None = None,
        in_row_offsets: mx.array | None = None,
        in_edge_ids: mx.array | None = None,
        kernel_row_offsets: mx.array | None = None,
        kernel_edge_ids: mx.array | None = None,
        kernel_offsets: tuple[Triple, ...] = (),
        out_coords: mx.array | None = None,
        n_in_capacity: int | None = None,
        n_out_capacity: int | None = None,
        n_kernels: int | None = None,
        source_coords: mx.array | None = None,
        source_active_rows: mx.array | None = None,
        target_coords: mx.array | None = None,
        target_active_rows: mx.array | None = None,
        stride: Triple = (1, 1, 1),
        padding: Triple = (0, 0, 0),
        kind: RelationKind = 'forward',
        implicit_gemm: RelationImplicitGemmView | None = None,
        sorted_implicit_gemm: RelationSortedImplicitGemmView | None = None,
    ) -> None:
        if counts is None:
            counts = mx.array(
                [
                    in_rows.shape[0],
                    0 if out_coords is None else out_coords.shape[0],
                ],
                dtype=mx.int32,
            )
        _validate_counts(counts)
        contract = SparseRelationContract(
            counts=counts,
            kernel_offsets=kernel_offsets,
            out_coords=out_coords,
            n_in_capacity=n_in_capacity,
            n_out_capacity=n_out_capacity,
            n_kernels=n_kernels,
            source_coords=source_coords,
            source_active_rows=source_active_rows,
            target_coords=target_coords,
            target_active_rows=target_active_rows,
            stride=stride,
            padding=padding,
            kind=kind,
        )
        if row_offsets is None:
            out_capacity = (
                0
                if contract.n_out_capacity is None
                else int(contract.n_out_capacity)
            )
            row_offsets = mx.array([0] * (out_capacity + 1), dtype=mx.int32)
        _validate_row_offsets(row_offsets)

        edges = RelationEdges(in_rows, out_rows, kernel_ids)
        if in_row_offsets is None:
            in_capacity = (
                0
                if contract.n_in_capacity is None
                else int(contract.n_in_capacity)
            )
            in_row_offsets = mx.array(
                [0] * (in_capacity + 1), dtype=mx.int32
            )
        if in_edge_ids is None:
            in_edge_ids = mx.array([0] * edges.capacity, dtype=mx.int32)
        if kernel_row_offsets is None:
            kernel_capacity = (
                0 if contract.n_kernels is None else int(contract.n_kernels)
            )
            kernel_row_offsets = mx.array(
                [0] * (kernel_capacity + 1), dtype=mx.int32
            )
        if kernel_edge_ids is None:
            kernel_edge_ids = mx.array([0] * edges.capacity, dtype=mx.int32)
        output_csr = RelationCSRView(row_offsets)
        input_csr = RelationCSRView(in_row_offsets, in_edge_ids)
        kernel_csr = RelationCSRView(kernel_row_offsets, kernel_edge_ids)
        if (
            contract.n_out_capacity is not None
            and int(row_offsets.shape[0])
            != int(contract.n_out_capacity) + 1
        ):
            raise ValueError(
                'row_offsets must have length n_out_capacity + 1.'
            )

        object.__setattr__(self, 'contract', contract)
        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'output_csr', output_csr)
        object.__setattr__(self, 'input_csr', input_csr)
        object.__setattr__(self, 'kernel_csr', kernel_csr)
        object.__setattr__(self, 'implicit_gemm', implicit_gemm)
        object.__setattr__(
            self, 'sorted_implicit_gemm', sorted_implicit_gemm
        )

    @property
    def edge_capacity(self) -> int:
        return self.edges.capacity

    @property
    def counts(self) -> mx.array:
        return self.contract.counts

    @property
    def edge_count(self) -> mx.array:
        return self.contract.edge_count

    @property
    def out_count(self) -> mx.array:
        return self.contract.out_count

    @property
    def kernel_offsets(self) -> tuple[Triple, ...]:
        return self.contract.kernel_offsets

    @property
    def out_coords(self) -> mx.array | None:
        return self.contract.out_coords

    @property
    def n_in_capacity(self) -> int | None:
        return self.contract.n_in_capacity

    @property
    def n_out_capacity(self) -> int | None:
        return self.contract.n_out_capacity

    @property
    def n_kernels(self) -> int | None:
        return self.contract.n_kernels

    @property
    def row_offsets(self) -> mx.array:
        return self.output_csr.row_offsets

    @property
    def in_row_offsets(self) -> mx.array:
        return self.input_csr.row_offsets

    @property
    def in_edge_ids(self) -> mx.array:
        edge_ids = self.input_csr.edge_ids
        if edge_ids is None:
            raise ValueError('input CSR view is missing edge ids.')
        return edge_ids

    @property
    def kernel_row_offsets(self) -> mx.array:
        return self.kernel_csr.row_offsets

    @property
    def kernel_edge_ids(self) -> mx.array:
        edge_ids = self.kernel_csr.edge_ids
        if edge_ids is None:
            raise ValueError('kernel CSR view is missing edge ids.')
        return edge_ids

    @property
    def out_view(self) -> RelationCSRView:
        return self.output_csr

    @property
    def in_view(self) -> RelationCSRView:
        return self.input_csr

    @property
    def kernel_view(self) -> RelationCSRView:
        return self.kernel_csr

    def require_implicit_gemm(self) -> RelationImplicitGemmView:
        view = self.implicit_gemm
        if view is not None:
            return view
        from mlx_lattice.core.coords.builders import (
            build_relation_implicit_gemm_view,
        )

        view = build_relation_implicit_gemm_view(self)
        object.__setattr__(self, 'implicit_gemm', view)
        return view

    def require_sorted_implicit_gemm(
        self,
    ) -> RelationSortedImplicitGemmView:
        sorted_view = self.sorted_implicit_gemm
        if sorted_view is not None:
            return sorted_view

        view = self.require_implicit_gemm()

        if view.row_masks.shape[1] != 1:
            raise ValueError(
                'sorted implicit GEMM view currently supports K <= 32.'
            )
        row_masks = view.row_masks[:, 0]
        sorted_rows = mx.argsort(row_masks).astype(mx.int32)
        sorted_out_in_map = view.out_in_map[sorted_rows]
        sorted_kv_out_in_map = mx.contiguous(
            mx.transpose(sorted_out_in_map)
        )
        tile_count = (int(view.out_in_map.shape[0]) + 63) // 64
        words = []
        for word in range(4):
            word_masks = mx.zeros((tile_count,), dtype=mx.int32)
            for row_offset in range(word * 16, (word + 1) * 16):
                rows = (
                    mx.arange(tile_count, dtype=mx.int32) * 64 + row_offset
                )
                valid = rows < int(view.out_in_map.shape[0])
                clipped = mx.minimum(
                    rows, int(view.out_in_map.shape[0]) - 1
                )
                masks = mx.where(valid, row_masks[sorted_rows[clipped]], 0)
                word_masks = mx.bitwise_or(word_masks, masks)
            words.append(word_masks)
        tile_masks = mx.reshape(mx.stack(words, axis=1), (-1,))
        sorted_view = RelationSortedImplicitGemmView(
            sorted_out_in_map=sorted_out_in_map,
            sorted_kv_out_in_map=sorted_kv_out_in_map,
            reorder_rows=sorted_rows,
            tile_masks=tile_masks,
        )
        object.__setattr__(self, 'sorted_implicit_gemm', sorted_view)
        return sorted_view


@dataclass(frozen=True, slots=True, init=False)
class NeighborRelation:
    edges: NeighborEdges
    distances: mx.array
    row_offsets: mx.array
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
        row_offsets: mx.array | None = None,
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
        if row_offsets is None:
            query_capacity = (
                0 if n_query_capacity is None else int(n_query_capacity)
            )
            row_offsets = mx.array(
                [0] * (query_capacity + 1), dtype=mx.int32
            )
        _validate_row_offsets(row_offsets)
        if n_query_capacity is not None and int(row_offsets.shape[0]) != (
            int(n_query_capacity) + 1
        ):
            raise ValueError(
                'row_offsets must have length n_query_capacity + 1.'
            )

        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'distances', distances)
        object.__setattr__(self, 'row_offsets', row_offsets)
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


def _validate_active_rows(value: mx.array, *, name: str) -> None:
    if value.shape != (1,) or value.dtype != mx.int32:
        raise ValueError(f'{name} must have shape (1,) and int32 dtype.')


def _validate_row_offsets(value: mx.array) -> None:
    if value.ndim != 1 or value.dtype != mx.int32:
        raise ValueError(
            'row_offsets must have shape (N + 1,) and int32 dtype.'
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
