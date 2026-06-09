from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice.core.coords.quantization import (
    VoxelReduction,
    sparse_quantize,
    voxelize_features,
)
from mlx_lattice.core.tensor import SparseTensor

__all__ = ['voxelize']


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
    voxel_feats = voxelize_features(
        feats,
        quantization,
        active_rows=active_rows,
        reduction=reduction,
    )
    return SparseTensor(
        quantization.coords,
        voxel_feats,
        stride=stride,
        active_rows=quantization.active_rows,
    )
