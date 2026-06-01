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
    downsample_coords as _downsample_coords,
)
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


def kernel_offsets(kernel_size: int | Sequence[int]) -> tuple[Triple, ...]:
    axes = []
    for size in triple(kernel_size, name='kernel_size'):
        if size <= 0:
            raise ValueError('kernel_size values must be positive.')
        if size % 2:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))
    return tuple(
        (int(x), int(y), int(z))
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


def build_kernel_map(
    coords: mx.array,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
) -> KernelMap:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('coords must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError('coords must be int32 or int64.')

    kernel = triple(kernel_size, name='kernel_size')
    step = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    if any(value <= 0 for value in step):
        raise ValueError('stride values must be positive.')
    if any(value < 0 for value in pad):
        raise ValueError('padding values must be non-negative.')

    (
        maps,
        sizes,
        kernels,
        residual_maps,
        residual_kernels,
        residual_offsets,
        out_coords,
        offset_values,
    ) = _build_kernel_map(coords, kernel, step, pad)
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


def _offsets_from_array(values: mx.array) -> tuple[Triple, ...]:
    rows = cast(list[list[int]], values.tolist())
    return tuple((int(row[0]), int(row[1]), int(row[2])) for row in rows)


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
