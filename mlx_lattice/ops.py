from __future__ import annotations

from mlx_lattice._ops.conv import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
)
from mlx_lattice._ops.feature import linear, relu, sigmoid
from mlx_lattice._ops.pool import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_pool,
    global_sum_pool,
    max_pool3d,
    pool3d,
)
from mlx_lattice._ops.tensor import cat, prune, sparse_collate, topk_rows
from mlx_lattice.point import downsample, intersection_coords, union_coords

spdownsample = downsample
sparse_coord_intersection = intersection_coords
sparse_coord_union = union_coords
sparse_conv3d = conv3d
average_pool3d = avg_pool3d
generic_sparse_conv_transpose3d = conv_transpose3d
generative_sparse_conv_transpose3d = generative_conv_transpose3d
sparse_pool3d = pool3d
sparse_max_pool3d = max_pool3d
sparse_avg_pool3d = avg_pool3d

__all__ = [
    'average_pool3d',
    'avg_pool3d',
    'cat',
    'conv3d',
    'conv_transpose3d',
    'downsample',
    'generative_conv_transpose3d',
    'generative_sparse_conv_transpose3d',
    'generic_sparse_conv_transpose3d',
    'global_avg_pool',
    'global_max_pool',
    'global_pool',
    'global_sum_pool',
    'intersection_coords',
    'linear',
    'max_pool3d',
    'pool3d',
    'prune',
    'relu',
    'sigmoid',
    'sparse_avg_pool3d',
    'sparse_collate',
    'sparse_conv3d',
    'sparse_coord_intersection',
    'sparse_coord_union',
    'sparse_max_pool3d',
    'sparse_pool3d',
    'spdownsample',
    'topk_rows',
    'union_coords',
]
