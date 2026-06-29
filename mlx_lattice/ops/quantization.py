from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import mlx.core as mx

from mlx_lattice.core.coords.quantization import (
    PointVoxelInterpolation,
    PointVoxelMap,
    SparseQuantization,
    VoxelReduction,
    _validate_reduction,
    build_point_voxel_map,
    interpolate_point_features,
    sparse_quantize,
)
from mlx_lattice.core.coords.quantization import (
    voxelize_features as _voxelize_features,
)
from mlx_lattice.core.tensor import SparseTensor
from mlx_lattice.core.types import triple

__all__ = [
    'build_point_voxel_map',
    'devoxelize',
    'interpolate_point_features',
    'voxelize',
    'voxelize_with_quantization',
]


@overload
def voxelize(
    points: mx.array,
    feats: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    active_rows: mx.array | None = None,
    reduction: VoxelReduction = 'mean',
    stride: int | Sequence[int] = 1,
) -> SparseTensor: ...


@overload
def voxelize(
    points: mx.array,
    feats: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    active_rows: mx.array | None = None,
    reduction: str,
    stride: int | Sequence[int] = 1,
) -> SparseTensor: ...


def voxelize(
    points: mx.array,
    feats: mx.array,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    active_rows: mx.array | None = None,
    reduction: str = 'mean',
    stride: int | Sequence[int] = 1,
) -> SparseTensor:
    """Quantize points and aggregate features into a sparse voxel tensor.

    ``points`` must have shape ``(N, 3)`` and ``float32`` dtype. ``feats`` must
    have shape ``(N, C)``. Optional ``batch_indices`` has shape ``(N,)`` and
    ``int32`` dtype. The returned tensor uses coordinates
    ``(batch, floor((point - origin) / voxel_size))`` and aggregated voxel
    features.
    """
    if feats.ndim != 2:
        raise ValueError('feats must have shape (N, C).')
    if points.shape[0] != feats.shape[0]:
        raise ValueError('points and feats must have matching rows.')

    quantization = sparse_quantize(
        points,
        voxel_size,
        batch_indices=batch_indices,
        origin=origin,
        active_rows=active_rows,
    )
    return voxelize_with_quantization(
        quantization,
        feats,
        active_rows=active_rows,
        reduction=_validate_reduction(reduction),
        stride=stride,
    )


@overload
def voxelize_with_quantization(
    quantization: SparseQuantization,
    feats: mx.array,
    *,
    active_rows: mx.array | None = None,
    reduction: VoxelReduction = 'mean',
    stride: int | Sequence[int] = 1,
    template: SparseTensor | None = None,
) -> SparseTensor: ...


@overload
def voxelize_with_quantization(
    quantization: SparseQuantization,
    feats: mx.array,
    *,
    active_rows: mx.array | None = None,
    reduction: str,
    stride: int | Sequence[int] = 1,
    template: SparseTensor | None = None,
) -> SparseTensor: ...


def voxelize_with_quantization(
    quantization: SparseQuantization,
    feats: mx.array,
    *,
    active_rows: mx.array | None = None,
    reduction: str = 'mean',
    stride: int | Sequence[int] = 1,
    template: SparseTensor | None = None,
) -> SparseTensor:
    """Apply precomputed sparse quantization metadata to feature rows.

    This is the split form of ``voxelize`` for pipelines that reuse point to
    voxel assignments across multiple feature tensors. If ``template`` is
    supplied, it must share the quantization coordinate and active-row arrays.
    """
    voxel_feats = _voxelize_features(
        feats,
        quantization,
        active_rows=active_rows,
        reduction=_validate_reduction(reduction),
    )
    if template is not None:
        if template.coords is not quantization.coords:
            raise ValueError(
                'template must use the quantization coordinate array.'
            )
        if template.active_rows is not quantization.active_rows:
            raise ValueError(
                'template must use the quantization active_rows array.'
            )
        return template.replace(feats=voxel_feats)

    return SparseTensor(
        quantization.coords,
        voxel_feats,
        stride=triple(stride, name='stride'),
        active_rows=quantization.active_rows,
    )


def devoxelize(
    points: mx.array,
    voxels: SparseTensor,
    voxel_size: float | Sequence[float] = 1.0,
    *,
    batch_indices: mx.array | None = None,
    point_active_rows: mx.array | None = None,
    origin: float | Sequence[float] = 0.0,
    interpolation: PointVoxelInterpolation = 'linear',
    point_voxel_map: PointVoxelMap | None = None,
) -> mx.array:
    """Sample sparse voxel features back onto dense point rows.

    If ``point_voxel_map`` is not supplied, the function builds one from point
    coordinates, voxel coordinates, voxel size, origin, batch indices, and the
    selected interpolation mode. The returned dense array has one row per point
    and one column per voxel feature channel.
    """
    point_map = point_voxel_map
    if point_map is None:
        point_map = build_point_voxel_map(
            points,
            voxels.coords,
            voxels.active_rows,
            voxel_size,
            batch_indices=batch_indices,
            point_active_rows=point_active_rows,
            origin=origin,
            interpolation=interpolation,
        )
    return interpolate_point_features(voxels.feats, point_map)
