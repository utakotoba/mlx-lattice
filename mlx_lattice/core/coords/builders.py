from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.relations import (
    KernelRelation,
    KernelSpec,
)
from mlx_lattice.core.types import Triple, triple

type NativeKernelRelation = tuple[
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
        n_in_rows=int(coords.shape[0]),
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
        n_in_rows=int(coords.shape[0]),
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
        n_in_rows=int(coords.shape[0]),
    )


# MARK: - views


def _kernel_relation_from_native(
    native: NativeKernelRelation,
    *,
    offsets: tuple[Triple, ...],
    n_in_rows: int,
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
        n_in_rows=n_in_rows,
        n_out_rows=int(out_coords.shape[0]),
        n_kernels=len(offsets),
    )


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')


def _active_rows(value: mx.array | None, coords: mx.array) -> mx.array:
    if value is not None:
        if value.shape != (1,) or value.dtype != mx.int32:
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )
        return value
    return mx.array([coords.shape[0]], dtype=mx.int32)
