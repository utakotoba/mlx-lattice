from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import mlx.core as mx

from mlx_lattice._native import max_pool3d_feats as _max_pool3d_feats
from mlx_lattice._native import pool3d_feats as _pool3d_feats
from mlx_lattice._ops.conv import conv3d
from mlx_lattice.point import KernelMap, kernel_offsets
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import triple


def pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    mode: str = 'sum',
    kernel_map: KernelMap | None = None,
) -> SparseTensor:
    kernel = triple(kernel_size, name='kernel_size')
    op_stride = triple(stride, name='stride')
    mapping = kernel_map or x.kernel_map(
        kernel_size=kernel, stride=op_stride
    )

    if mode == 'avg':
        summed = pool3d(
            x,
            kernel_size=kernel,
            stride=op_stride,
            mode='sum',
            kernel_map=mapping,
        )
        counts = pool3d(
            x.replace(feats=mx.ones((x.n_points, 1), dtype=x.feats.dtype)),
            kernel_size=kernel,
            stride=op_stride,
            mode='sum',
            kernel_map=mapping,
        )
        return summed.replace(feats=summed.feats / counts.feats)
    if mode == 'max':
        out_rows = int(mapping.out_coords.shape[0])
        if (
            mapping.center >= 0
            and op_stride == (1, 1, 1)
            and out_rows == x.n_points
        ):
            feats = _max_pool3d_feats(
                x.feats,
                mapping.residual_maps,
                mapping.residual_kernels,
                mapping.residual_offsets,
                out_rows,
            )
            return _pooled_tensor(
                x, mapping.out_coords, mx.maximum(feats, x.feats), op_stride
            )
        if mapping.center >= 0:
            maps, kernels, offsets = _row_view_from_maps(
                mapping.maps, mapping.kernels, out_rows
            )
        else:
            maps = mapping.residual_maps
            kernels = mapping.residual_kernels
            offsets = mapping.residual_offsets
        feats = _max_pool3d_feats(
            x.feats,
            maps,
            kernels,
            offsets,
            out_rows,
        )
        return _pooled_tensor(x, mapping.out_coords, feats, op_stride)
    if mode != 'sum':
        raise ValueError("pool mode must be 'sum', 'max', or 'avg'.")

    if mapping.center >= 0:
        volume = len(kernel_offsets(kernel))
        weight = mx.broadcast_to(
            mx.eye(x.channels, dtype=x.feats.dtype),
            (volume, x.channels, x.channels),
        )
        return conv3d(
            x,
            weight,
            kernel_size=kernel,
            stride=op_stride,
            kernel_map=mapping,
        )

    feats = _pool3d_feats(
        x.feats,
        mapping.residual_maps,
        mapping.residual_kernels,
        mapping.residual_offsets,
        int(mapping.out_coords.shape[0]),
    )
    return _pooled_tensor(x, mapping.out_coords, feats, op_stride)


def max_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    kernel_map: KernelMap | None = None,
) -> SparseTensor:
    return pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        mode='max',
        kernel_map=kernel_map,
    )


def avg_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    kernel_map: KernelMap | None = None,
) -> SparseTensor:
    return pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        mode='avg',
        kernel_map=kernel_map,
    )


def global_pool(
    x: SparseTensor,
    *,
    mode: str = 'avg',
) -> SparseTensor:
    pooled = []
    coords = []
    if x.batch_counts is None:
        for rows in x.batch_rows:
            if rows.shape[0] == 0:
                continue
            batch = int(mx.take(x.coords[:, 0], rows[0]).item())
            pooled.append(
                _global_pool_feats(mx.take(x.feats, rows, axis=0), mode)
            )
            coords.append([batch, 0, 0, 0])
    else:
        start = 0
        for batch, row_count in enumerate(x.batch_counts):
            stop = start + int(row_count)
            if stop > x.n_points:
                raise ValueError(
                    'batch row counts exceed sparse tensor row count.'
                )
            pooled.append(_global_pool_feats(x.feats[start:stop], mode))
            coords.append([batch, 0, 0, 0])
            start = stop
        if start != x.n_points:
            raise ValueError('batch row counts must cover all sparse rows.')

    if not pooled:
        out_coords = mx.array([], dtype=x.coords.dtype).reshape((0, 4))
        out_feats = mx.array([], dtype=x.feats.dtype).reshape(
            (0, x.channels)
        )
    else:
        out_coords = mx.array(coords, dtype=x.coords.dtype)
        out_feats = mx.concatenate(pooled, axis=0)
    return SparseTensor(out_coords, out_feats, x.stride)


def _global_pool_feats(feats: mx.array, mode: str) -> mx.array:
    if feats.shape[0] == 0:
        return mx.zeros((1, feats.shape[1]), dtype=feats.dtype)
    if mode == 'sum':
        return mx.sum(feats, axis=0, keepdims=True)
    if mode == 'avg':
        return mx.mean(feats, axis=0, keepdims=True)
    if mode == 'max':
        return mx.max(feats, axis=0, keepdims=True)
    raise ValueError("pool mode must be 'sum', 'max', or 'avg'.")


def global_sum_pool(
    x: SparseTensor,
) -> SparseTensor:
    return global_pool(x, mode='sum')


def global_avg_pool(
    x: SparseTensor,
) -> SparseTensor:
    return global_pool(x, mode='avg')


def global_max_pool(
    x: SparseTensor,
) -> SparseTensor:
    return global_pool(x, mode='max')


def _pooled_tensor(
    x: SparseTensor,
    coords: mx.array,
    feats: mx.array,
    stride: tuple[int, int, int],
) -> SparseTensor:
    out_stride = tuple(a * b for a, b in zip(x.stride, stride, strict=True))
    return SparseTensor(
        coords,
        feats,
        out_stride,
        coord_manager=x.coord_manager,
    )


def _row_view_from_maps(
    maps: mx.array,
    kernels: mx.array,
    rows: int,
) -> tuple[mx.array, mx.array, mx.array]:
    map_rows = cast(list[list[int]], maps.tolist())
    kernel_rows = cast(list[int], kernels.tolist())
    grouped: list[list[tuple[int, int]]] = [[] for _ in range(rows)]
    for pair, row in zip(map_rows, kernel_rows, strict=True):
        grouped[int(pair[1])].append((int(pair[0]), int(row)))

    out_maps = []
    out_kernels = []
    offsets = [0]
    for out_row, pairs in enumerate(grouped):
        for in_row, kernel in pairs:
            out_maps.append([in_row, out_row])
            out_kernels.append(kernel)
        offsets.append(len(out_kernels))
    return (
        mx.array(out_maps, dtype=mx.int32).reshape((-1, 2)),
        mx.array(out_kernels, dtype=mx.int32),
        mx.array(offsets, dtype=mx.int32),
    )
