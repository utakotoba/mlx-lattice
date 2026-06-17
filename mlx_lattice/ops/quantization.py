from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import mlx.core as mx

from mlx_lattice.core.coords.quantization import (
    SparseQuantization,
    VoxelReduction,
    _validate_reduction,
    sparse_quantize,
)
from mlx_lattice.core.coords.quantization import (
    voxelize_features as _voxelize_features,
)
from mlx_lattice.core.tensor import SparseTensor
from mlx_lattice.core.types import triple

__all__ = ['voxelize', 'voxelize_with_quantization']


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
    """Apply a precomputed native point-to-voxel map to feature rows."""
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
