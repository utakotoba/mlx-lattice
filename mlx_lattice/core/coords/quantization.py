from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coords

type VoxelReduction = Literal['sum', 'mean']
type PointVoxelInterpolation = Literal['nearest', 'linear']


@dataclass(frozen=True, slots=True)
class SparseQuantization:
    """Sparse voxel coordinates plus point-to-voxel metadata.

    ``coords`` stores unique voxel coordinates in ``(batch, x, y, z)`` order.
    ``inverse_rows`` maps each input point row to its voxel row, and ``counts``
    stores the number of active point rows accumulated into each voxel.
    """

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


@dataclass(frozen=True, slots=True)
class PointVoxelMap:
    """Fixed-width point-to-voxel interpolation rows and weights.

    ``rows`` and ``weights`` both have shape ``(N, 8)``. Linear interpolation
    may use up to eight neighboring voxel rows per point; nearest interpolation
    uses one non-zero contribution.
    """

    rows: mx.array
    weights: mx.array

    def __post_init__(self) -> None:
        if (
            self.rows.ndim != 2
            or self.rows.shape[1] != 8
            or self.rows.dtype != mx.int32
        ):
            raise ValueError('rows must have shape (N, 8) and int32 dtype.')
        if self.weights.shape != self.rows.shape:
            raise ValueError('weights must have the same shape as rows.')
        if self.weights.dtype != mx.float32:
            raise ValueError('weights must be float32.')

    @property
    def point_count(self) -> int:
        return int(self.rows.shape[0])


def sparse_quantize(
    points: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    active_rows: mx.array | None = None,
) -> SparseQuantization:
    """Voxelize floating-point points into sparse integer coordinates.

    Points have shape ``(N, 3)`` and dtype ``float32``. Optional
    ``batch_indices`` assign points to batches; omitted batches default to
    zero. The result includes voxel coordinates, active row count, inverse
    point-to-voxel rows, and per-voxel counts.
    """
    _validate_points(points)
    batches = _batch_indices(batch_indices, points.shape[0])
    point_rows = _active_rows(active_rows, points.shape[0])
    native = ext.sparse_quantize(
        points,
        batches,
        point_rows,
        _float_triple(voxel_size, 'voxel_size'),
        _float_triple(origin, 'origin'),
    )
    return SparseQuantization(*native)


def voxelize_features(
    feats: mx.array,
    quantization: SparseQuantization,
    *,
    active_rows: mx.array | None = None,
    reduction: VoxelReduction = 'mean',
) -> mx.array:
    """Aggregate point features into voxels using sparse quantization data.

    ``feats`` must be ``float32`` with one row per original point. ``sum``
    accumulates point rows directly; ``mean`` divides by each voxel count.
    """
    if feats.ndim != 2:
        raise ValueError('feats must have shape (N, C).')
    if feats.dtype != mx.float32:
        raise ValueError('voxelize currently supports float32 features.')
    if feats.shape[0] != quantization.capacity:
        raise ValueError('feats and quantization metadata must share rows.')
    point_rows = _active_rows(active_rows, feats.shape[0])
    return ext.voxelize_features(
        feats,
        quantization.inverse_rows,
        quantization.counts,
        point_rows,
        _validate_reduction(reduction),
    )


def build_point_voxel_map(
    points: mx.array,
    voxel_coords: mx.array,
    voxel_active_rows: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    point_active_rows: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    interpolation: PointVoxelInterpolation = 'linear',
) -> PointVoxelMap:
    """Build fixed-width interpolation rows from points to voxel centers.

    The map can be reused to sample multiple voxel feature arrays as long as
    point geometry, batch indices, voxel coordinates, voxel size, and origin
    are unchanged.
    """
    _validate_points(points)
    validate_coords(voxel_coords)
    if voxel_coords.dtype != mx.int32:
        raise ValueError('voxel_coords must be int32.')
    batches = _batch_indices(batch_indices, points.shape[0])
    point_rows = _active_rows(point_active_rows, points.shape[0])
    if (
        voxel_active_rows.shape != (1,)
        or voxel_active_rows.dtype != mx.int32
    ):
        raise ValueError(
            'voxel_active_rows must have shape (1,) and int32 dtype.'
        )
    native = ext.build_point_voxel_map(
        points,
        batches,
        point_rows,
        voxel_coords,
        voxel_active_rows,
        _float_triple(voxel_size, 'voxel_size'),
        _float_triple(origin, 'origin'),
        _validate_interpolation(interpolation),
    )
    return PointVoxelMap(*native)


def interpolate_point_features(
    voxel_feats: mx.array,
    point_voxel_map: PointVoxelMap,
) -> mx.array:
    """Interpolate voxel features back to point rows.

    ``voxel_feats`` must be ``float32`` with shape ``(N_voxels, C)``. The
    returned dense point feature array has shape ``(N_points, C)``.
    """
    if voxel_feats.ndim != 2:
        raise ValueError('voxel_feats must have shape (N, C).')
    if voxel_feats.dtype != mx.float32:
        raise ValueError('point interpolation currently supports float32.')
    return ext.interpolate_point_features(
        voxel_feats,
        point_voxel_map.rows,
        point_voxel_map.weights,
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


def _validate_interpolation(value: str) -> PointVoxelInterpolation:
    if value == 'nearest':
        return 'nearest'
    if value == 'linear':
        return 'linear'
    raise ValueError("interpolation must be 'nearest' or 'linear'.")
