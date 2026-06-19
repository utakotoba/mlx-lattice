from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_lattice._native import native
from mlx_lattice.core.coords.validation import validate_coords

type VoxelReduction = Literal['sum', 'mean']


@dataclass(frozen=True, slots=True)
class SparseQuantization:
    """Sparse voxel coordinates plus point-to-voxel metadata."""

    coords: mx.array
    active_rows: mx.array
    inverse_rows: mx.array
    counts: mx.array

    def __post_init__(self) -> None:
        validate_coords(self.coords)
        if self.coords.dtype != mx.int32:
            raise ValueError('quantized coords must be int32.')
        if (
            self.active_rows.shape != (1,)
            or self.active_rows.dtype != mx.int32
        ):
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )
        if (
            self.inverse_rows.shape != (self.coords.shape[0],)
            or self.inverse_rows.dtype != mx.int32
        ):
            raise ValueError(
                'inverse_rows must have shape (N,) and int32 dtype.'
            )
        if (
            self.counts.shape != (self.coords.shape[0],)
            or self.counts.dtype != mx.int32
        ):
            raise ValueError('counts must have shape (N,) and int32 dtype.')

    @property
    def capacity(self) -> int:
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        return self.active_rows


def sparse_quantize(
    points: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    active_rows: mx.array | None = None,
) -> SparseQuantization:
    _validate_points(points)
    batches = _batch_indices(batch_indices, points.shape[0])
    point_rows = _active_rows(active_rows, points.shape[0])
    voxel = _float_triple(voxel_size, 'voxel_size')
    offset = _float_triple(origin, 'origin')
    out = native.sparse_quantize(
        points,
        batches,
        point_rows,
        voxel,
        offset,
    )
    return SparseQuantization(*out)


def voxelize_features(
    feats: mx.array,
    quantization: SparseQuantization,
    *,
    active_rows: mx.array | None = None,
    reduction: VoxelReduction = 'mean',
) -> mx.array:
    if feats.ndim != 2:
        raise ValueError('feats must have shape (N, C).')
    if feats.dtype != mx.float32:
        raise ValueError('voxelize currently supports float32 features.')
    if feats.shape[0] != quantization.capacity:
        raise ValueError('feats and quantization metadata must share rows.')
    point_rows = _active_rows(active_rows, feats.shape[0])
    reduce = _validate_reduction(reduction)
    return native.voxelize_features(
        feats,
        quantization.inverse_rows,
        quantization.counts,
        point_rows,
        reduce,
    )


def _validate_points(points: mx.array) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if points.dtype != mx.float32:
        raise ValueError('points must be float32.')


def _batch_indices(value: mx.array | None, rows: int) -> mx.array:
    if value is None:
        return mx.zeros((rows,), dtype=mx.int32)
    if value.shape != (rows,) or value.dtype != mx.int32:
        raise ValueError(
            'batch_indices must have shape (N,) and int32 dtype.'
        )
    return value


def _active_rows(value: mx.array | None, rows: int) -> mx.array:
    if value is None:
        return mx.array([rows], dtype=mx.int32)
    if value.shape != (1,) or value.dtype != mx.int32:
        raise ValueError(
            'active_rows must have shape (1,) and int32 dtype.'
        )
    return value


def _float_triple(
    value: float | Sequence[float],
    name: str,
) -> tuple[float, float, float]:
    if isinstance(value, int | float):
        return (float(value), float(value), float(value))
    else:
        raw = tuple(float(item) for item in value)
        if len(raw) != 3:
            raise ValueError(f'{name} must contain exactly 3 values.')
        values = (raw[0], raw[1], raw[2])
    if name == 'voxel_size' and any(item <= 0.0 for item in values):
        raise ValueError('voxel_size values must be positive.')
    return values


def _validate_reduction(value: str) -> VoxelReduction:
    if value == 'sum':
        return 'sum'
    if value == 'mean':
        return 'mean'
    raise ValueError("reduction must be 'sum' or 'mean'.")
