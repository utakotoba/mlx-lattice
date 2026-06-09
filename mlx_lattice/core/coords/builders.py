from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.relations import (
    KernelRelation,
    KernelSpec,
    NeighborRelation,
)
from mlx_lattice.core.types import Triple, triple

type NativeKernelRelation = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]
type NativeNeighborRelation = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]


def kernel_offsets(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    kernel = triple(kernel_size, name='kernel_size')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(rate, 'dilation')

    axes = []
    for size in kernel:
        if size % 2 == 1:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))

    return tuple(
        (int(x * rate[0]), int(y * rate[1]), int(z * rate[2]))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def build_kernel_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    validate_coords(coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    offsets = kernel_offsets(spec.size, spec.dilation)
    native = ext.build_kernel_relation(
        coords,
        _active_rows(active_rows, coords),
        spec.size,
        spec.stride,
        spec.padding,
        spec.dilation,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
    )


def build_generative_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelRelation:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')

    offsets = kernel_offsets(kernel)
    native = ext.build_generative_relation(
        coords,
        _active_rows(active_rows, coords),
        kernel,
        step,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
    )


def build_transposed_kernel_relation(
    coords: mx.array,
    active_rows: mx.array | None = None,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')
    _require_nonnegative(pad, 'padding')
    _require_positive(rate, 'dilation')

    offsets = kernel_offsets(kernel, rate)
    native = ext.build_transposed_kernel_relation(
        coords,
        _active_rows(active_rows, coords),
        kernel,
        step,
        pad,
        rate,
    )
    return _kernel_relation_from_native(
        native,
        offsets=offsets,
        in_capacity=int(coords.shape[0]),
    )


def build_knn_relation(
    source_coords: mx.array,
    query_coords: mx.array | None = None,
    *,
    source_active_rows: mx.array | None = None,
    query_active_rows: mx.array | None = None,
    k: int,
) -> NeighborRelation:
    query_coords = source_coords if query_coords is None else query_coords
    source_active_rows = _active_rows(source_active_rows, source_coords)
    query_active_rows = (
        source_active_rows
        if query_active_rows is None and query_coords is source_coords
        else _active_rows(query_active_rows, query_coords)
    )
    validate_coords(source_coords)
    validate_coords(query_coords)
    _require_matching_coord_dtype(source_coords, query_coords)
    neighbor_count = _positive_int(k, 'k')
    native = ext.build_knn_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        neighbor_count,
    )
    return _neighbor_relation_from_native(
        native,
        query_capacity=int(query_coords.shape[0]),
        source_capacity=int(source_coords.shape[0]),
        max_neighbors=neighbor_count,
    )


def build_radius_relation(
    source_coords: mx.array,
    query_coords: mx.array | None = None,
    *,
    source_active_rows: mx.array | None = None,
    query_active_rows: mx.array | None = None,
    radius: float,
    max_neighbors: int | None = None,
) -> NeighborRelation:
    query_coords = source_coords if query_coords is None else query_coords
    source_active_rows = _active_rows(source_active_rows, source_coords)
    query_active_rows = (
        source_active_rows
        if query_active_rows is None and query_coords is source_coords
        else _active_rows(query_active_rows, query_coords)
    )
    validate_coords(source_coords)
    validate_coords(query_coords)
    _require_matching_coord_dtype(source_coords, query_coords)
    radius_value = _nonnegative_float(radius, 'radius')
    neighbor_count = (
        int(source_coords.shape[0])
        if max_neighbors is None
        else _positive_int(max_neighbors, 'max_neighbors')
    )
    native = ext.build_radius_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        radius_value,
        neighbor_count,
    )
    return _neighbor_relation_from_native(
        native,
        query_capacity=int(query_coords.shape[0]),
        source_capacity=int(source_coords.shape[0]),
        max_neighbors=neighbor_count,
    )


# MARK: - views


def _kernel_relation_from_native(
    native: NativeKernelRelation,
    *,
    offsets: tuple[Triple, ...],
    in_capacity: int,
) -> KernelRelation:
    (
        in_rows,
        out_rows,
        kernel_ids,
        out_coords,
        counts,
    ) = native
    return KernelRelation(
        in_rows,
        out_rows,
        kernel_ids,
        counts=counts,
        kernel_offsets=offsets,
        out_coords=out_coords,
        n_in_capacity=in_capacity,
        n_out_capacity=int(out_coords.shape[0]),
        n_kernels=len(offsets),
    )


def _neighbor_relation_from_native(
    native: NativeNeighborRelation,
    *,
    query_capacity: int,
    source_capacity: int,
    max_neighbors: int,
) -> NeighborRelation:
    query_rows, source_rows, neighbor_ids, distances, counts = native
    return NeighborRelation(
        query_rows,
        source_rows,
        neighbor_ids,
        distances,
        counts=counts,
        n_query_capacity=query_capacity,
        n_source_capacity=source_capacity,
        max_neighbors=max_neighbors,
    )


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')


def _require_matching_coord_dtype(lhs: mx.array, rhs: mx.array) -> None:
    if lhs.dtype != rhs.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')


def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f'{name} must be positive.')
    return out


def _nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f'{name} must be non-negative.')
    return out


def _active_rows(value: mx.array | None, coords: mx.array) -> mx.array:
    if value is not None:
        if value.shape != (1,) or value.dtype != mx.int32:
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )
        return value
    return mx.array([coords.shape[0]], dtype=mx.int32)
