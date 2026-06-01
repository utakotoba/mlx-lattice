from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from mlx_lattice._native import (
    build_generative_map as _build_generative_map,
)
from mlx_lattice._native import (
    build_kernel_map as _build_kernel_map,
)
from mlx_lattice._native import (
    build_transposed_kernel_map as _build_transposed_kernel_map,
)
from mlx_lattice._native import (
    downsample_coords as _downsample_coords,
)
from mlx_lattice._native import (
    intersection_coords as _intersection_coords,
)
from mlx_lattice._native import lookup_coords as _lookup_coords
from mlx_lattice._native import union_coords as _union_coords
from mlx_lattice.types import Triple, triple


@dataclass(frozen=True)
class KernelMap:
    maps: mx.array
    sizes: mx.array
    kernels: mx.array
    residual_maps: mx.array
    residual_kernels: mx.array
    residual_offsets: mx.array
    out_coords: mx.array
    offsets: tuple[Triple, ...]

    @property
    def center(self) -> int:
        try:
            return self.offsets.index((0, 0, 0))
        except ValueError:
            return -1


def kernel_offsets(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> tuple[Triple, ...]:
    axes = []
    rates = triple(dilation, name='dilation')
    if any(value <= 0 for value in rates):
        raise ValueError('dilation values must be positive.')
    for size in triple(kernel_size, name='kernel_size'):
        if size <= 0:
            raise ValueError('kernel_size values must be positive.')
        if size % 2:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))
    return tuple(
        (int(x * rates[0]), int(y * rates[1]), int(z * rates[2]))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def downsample(
    coords: mx.array,
    stride: int | Sequence[int] = 2,
) -> mx.array:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    step = triple(stride, name='stride')
    if any(value <= 0 for value in step):
        raise ValueError('stride values must be positive.')
    return _downsample_coords(coords, step)


def union_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    _validate_coord_pair(lhs, rhs)
    return _place_coord_array(lhs, _union_coords(lhs, rhs))


def intersection_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    _validate_coord_pair(lhs, rhs)
    return _place_coord_array(lhs, _intersection_coords(lhs, rhs))


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    _validate_coord_pair(coords, queries)
    return _place_coord_array(coords, _lookup_coords(coords, queries))


def inverse_map(source: mx.array, target: mx.array) -> mx.array:
    return lookup_coords(source, target)


def build_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    if any(value <= 0 for value in step):
        raise ValueError('stride values must be positive.')
    if any(value < 0 for value in pad):
        raise ValueError('padding values must be non-negative.')
    if any(value <= 0 for value in rate):
        raise ValueError('dilation values must be positive.')

    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
        offset_values,
    ) = _build_kernel_map(coords, kernel, step, pad, rate)
    offsets = _offsets_from_array(offset_values)
    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    ) = _place_cached_map_arrays(
        coords,
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    )
    return KernelMap(
        maps=maps,
        sizes=sizes,
        kernels=kernels,
        residual_maps=residual_maps,
        residual_kernels=residual_kernels,
        residual_offsets=residual_offsets,
        out_coords=out_coords,
        offsets=offsets,
    )


def build_generative_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelMap:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    if any(value <= 0 for value in step):
        raise ValueError('stride values must be positive.')
    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
        offset_values,
    ) = _build_generative_map(coords, kernel, step)
    offsets = _offsets_from_array(offset_values)
    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    ) = _place_cached_map_arrays(
        coords,
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    )
    return KernelMap(
        maps=maps,
        sizes=sizes,
        kernels=kernels,
        residual_maps=residual_maps,
        residual_kernels=residual_kernels,
        residual_offsets=residual_offsets,
        out_coords=out_coords,
        offsets=offsets,
    )


def build_transposed_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelMap:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    if any(value <= 0 for value in step):
        raise ValueError('stride values must be positive.')
    if any(value < 0 for value in pad):
        raise ValueError('padding values must be non-negative.')
    if any(value <= 0 for value in rate):
        raise ValueError('dilation values must be positive.')
    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
        offset_values,
    ) = _build_transposed_kernel_map(coords, kernel, step, pad, rate)
    offsets = _offsets_from_array(offset_values)
    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    ) = _place_cached_map_arrays(
        coords,
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
    )
    return KernelMap(
        maps=maps,
        sizes=sizes,
        kernels=kernels,
        residual_maps=residual_maps,
        residual_kernels=residual_kernels,
        residual_offsets=residual_offsets,
        out_coords=out_coords,
        offsets=offsets,
    )


def _offsets_from_array(values: mx.array) -> tuple[Triple, ...]:
    rows = cast(list[list[int]], values.tolist())
    return tuple((int(row[0]), int(row[1]), int(row[2])) for row in rows)


def _validate_coords(coords: mx.array, *, name: str = 'coords') -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_coord_pair(lhs: mx.array, rhs: mx.array) -> None:
    _validate_coords(lhs, name='lhs')
    _validate_coords(rhs, name='rhs')
    if lhs.dtype != rhs.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')


def _place_coord_array(coords: mx.array, values: mx.array) -> mx.array:
    return _place_cached_map_arrays(coords, values)[0]


def _place_cached_map_arrays(
    coords: mx.array,
    *arrays: mx.array,
) -> tuple[mx.array, ...]:
    gpu_available = mx.metal.is_available() or mx.cuda.is_available()
    if (
        coords.dtype == mx.int32
        and gpu_available
        and mx.default_device() == mx.Device(mx.gpu)
    ):
        device = mx.default_device()
        return tuple(
            mx.contiguous(array, stream=device) for array in arrays
        )
    return arrays
