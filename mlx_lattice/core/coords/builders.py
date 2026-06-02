from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import (
    make_i32_array,
    validate_coords,
)
from mlx_lattice.core.maps import (
    InputCsrView,
    KernelBucketView,
    KernelMap,
    KernelSpec,
    OutputCsrView,
)
from mlx_lattice.core.types import Triple, triple

type NativeKernelMap = tuple[
    mx.array, mx.array, mx.array, mx.array, mx.array
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


def build_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    validate_coords(coords)
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return _kernel_map_from_native(
        ext.build_kernel_map(
            coords,
            spec.size,
            spec.stride,
            spec.padding,
            spec.dilation,
        ),
        n_in_rows=int(coords.shape[0]),
    )


def build_generative_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelMap:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')

    return _kernel_map_from_native(
        ext.build_generative_map(
            coords,
            kernel,
            step,
        ),
        n_in_rows=int(coords.shape[0]),
    )


def build_transposed_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    validate_coords(coords)
    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    _require_positive(kernel, 'kernel_size')
    _require_positive(step, 'stride')
    _require_nonnegative(pad, 'padding')
    _require_positive(rate, 'dilation')

    return _kernel_map_from_native(
        ext.build_transposed_kernel_map(
            coords,
            kernel,
            step,
            pad,
            rate,
        ),
        n_in_rows=int(coords.shape[0]),
    )


# MARK: - views


def _kernel_map_from_native(
    native: NativeKernelMap,
    *,
    n_in_rows: int,
) -> KernelMap:
    in_rows, out_rows, kernel_ids, out_coords, offset_values = native
    offsets = _offsets_from_array(offset_values)
    edges = _edges_from_arrays(in_rows, out_rows, kernel_ids)
    return _kernel_map_from_edges(
        edges,
        in_rows=in_rows,
        out_rows=out_rows,
        kernel_ids=kernel_ids,
        out_coords=out_coords,
        kernel_offsets=offsets,
        n_in_rows=n_in_rows,
        n_out_rows=int(out_coords.shape[0]),
    )


def _kernel_map_from_edges(
    edges: Sequence[tuple[int, int, int]],
    *,
    in_rows: mx.array | None = None,
    out_rows: mx.array | None = None,
    kernel_ids: mx.array | None = None,
    out_coords: mx.array,
    kernel_offsets: tuple[Triple, ...],
    n_in_rows: int,
    n_out_rows: int,
) -> KernelMap:
    in_row_values = [edge[0] for edge in edges]
    out_row_values = [edge[1] for edge in edges]
    kernel_id_values = [edge[2] for edge in edges]
    return KernelMap(
        make_i32_array(in_row_values) if in_rows is None else in_rows,
        make_i32_array(out_row_values) if out_rows is None else out_rows,
        make_i32_array(kernel_id_values)
        if kernel_ids is None
        else kernel_ids,
        kernel_offsets=kernel_offsets,
        out_coords=out_coords,
        output_csr=_output_csr_view(edges, n_out_rows),
        kernel_buckets=_kernel_bucket_view(edges, len(kernel_offsets)),
        input_csr=_input_csr_view(edges, n_in_rows),
        n_in_rows=n_in_rows,
        n_out_rows=n_out_rows,
        n_kernels=len(kernel_offsets),
    )


def _output_csr_view(
    edges: Sequence[tuple[int, int, int]],
    n_out_rows: int,
) -> OutputCsrView:
    grouped = _group_by(edges, key=lambda edge: edge[1], rows=n_out_rows)
    return OutputCsrView(
        offsets=make_i32_array(_offsets(grouped)),
        in_rows=make_i32_array(
            edge[0] for bucket in grouped for edge in bucket
        ),
        kernel_ids=make_i32_array(
            edge[2] for bucket in grouped for edge in bucket
        ),
    )


def _kernel_bucket_view(
    edges: Sequence[tuple[int, int, int]],
    n_kernels: int,
) -> KernelBucketView:
    grouped = _group_by(edges, key=lambda edge: edge[2], rows=n_kernels)
    return KernelBucketView(
        offsets=make_i32_array(_offsets(grouped)),
        in_rows=make_i32_array(
            edge[0] for bucket in grouped for edge in bucket
        ),
        out_rows=make_i32_array(
            edge[1] for bucket in grouped for edge in bucket
        ),
    )


def _input_csr_view(
    edges: Sequence[tuple[int, int, int]],
    n_in_rows: int,
) -> InputCsrView:
    grouped = _group_by(edges, key=lambda edge: edge[0], rows=n_in_rows)
    return InputCsrView(
        offsets=make_i32_array(_offsets(grouped)),
        out_rows=make_i32_array(
            edge[1] for bucket in grouped for edge in bucket
        ),
        kernel_ids=make_i32_array(
            edge[2] for bucket in grouped for edge in bucket
        ),
    )


def _group_by(
    edges: Sequence[tuple[int, int, int]],
    *,
    key: Callable[[tuple[int, int, int]], int],
    rows: int,
) -> list[list[tuple[int, int, int]]]:
    grouped: list[list[tuple[int, int, int]]] = [[] for _ in range(rows)]
    for edge in edges:
        grouped[int(key(edge))].append(edge)
    return grouped


def _offsets(grouped: Sequence[Sequence[object]]) -> list[int]:
    out = [0]
    total = 0
    for bucket in grouped:
        total += len(bucket)
        out.append(total)
    return out


# MARK: - arrays


def _edges_from_arrays(
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
) -> list[tuple[int, int, int]]:
    in_values = cast(list[int], in_rows.tolist())
    out_values = cast(list[int], out_rows.tolist())
    kernel_values = cast(list[int], kernel_ids.tolist())
    return [
        (int(in_row), int(out_row), int(kernel_id))
        for in_row, out_row, kernel_id in zip(
            in_values,
            out_values,
            kernel_values,
            strict=True,
        )
    ]


def _offsets_from_array(values: mx.array) -> tuple[Triple, ...]:
    rows = cast(list[list[int]], values.tolist())
    return tuple((int(row[0]), int(row[1]), int(row[2])) for row in rows)


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')
