from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import mlx.core as mx

from mlx_lattice.core.tensor import SparseTensor
from mlx_lattice.core.types import triple


def sparse_collate(
    coords: Sequence[mx.array],
    feats: Sequence[mx.array],
    *,
    stride: int | Sequence[int] = 1,
) -> SparseTensor:
    if len(coords) != len(feats):
        raise ValueError('coords and feats batch sizes must match.')
    if not coords:
        raise ValueError('expected at least one sparse tensor batch.')

    batched_coords = []
    batched_feats = []
    coord_dtype = coords[0].dtype
    for batch, (coord_rows, feat_rows) in enumerate(
        zip(coords, feats, strict=True)
    ):
        if coord_rows.ndim != 2 or coord_rows.shape[1] != 3:
            raise ValueError('collated coords must have shape (N, 3).')
        if coord_rows.dtype not in (mx.int32, mx.int64):
            raise ValueError('collated coords must be int32 or int64.')
        if coord_rows.dtype != coord_dtype:
            raise ValueError('collated coords must share a dtype.')
        if feat_rows.ndim != 2:
            raise ValueError('collated feats must have shape (N, C).')
        if coord_rows.shape[0] != feat_rows.shape[0]:
            raise ValueError(
                'collated coords and feats must have matching rows.'
            )

        batch_col = mx.full(
            (coord_rows.shape[0], 1), batch, dtype=coord_rows.dtype
        )
        batched_coords.append(
            mx.concatenate([batch_col, coord_rows], axis=1)
        )
        batched_feats.append(feat_rows)

    return SparseTensor(
        mx.concatenate(batched_coords, axis=0),
        mx.concatenate(batched_feats, axis=0),
        stride=triple(stride, name='stride'),
        batch_counts=tuple(int(values.shape[0]) for values in coords),
    )


def cat(tensors: Sequence[SparseTensor]) -> SparseTensor:
    if not tensors:
        raise ValueError('expected at least one sparse tensor.')

    first = tensors[0]
    for tensor in tensors[1:]:
        if not first.same_coords(tensor):
            raise ValueError('sparse tensor coordinates must match.')
    return first.replace(
        feats=mx.concatenate([tensor.feats for tensor in tensors], axis=1)
    )


def prune(x: SparseTensor, rows: mx.array) -> SparseTensor:
    if rows.ndim != 1:
        raise ValueError('rows must have shape (M,).')
    rows = rows.astype(mx.int32)
    return SparseTensor(
        mx.take(x.coords, rows, axis=0),
        mx.take(x.feats, rows, axis=0),
        x.stride,
        coord_manager=x.coord_manager,
    )


def prune_mask(x: SparseTensor, mask: mx.array) -> SparseTensor:
    if mask.ndim != 1:
        raise ValueError('mask must have shape (N,).')
    if mask.shape[0] != x.capacity:
        raise ValueError('mask must match the sparse tensor capacity.')
    if mask.dtype != mx.bool_:
        raise ValueError('mask must be boolean.')

    ordering = mx.argsort(mask.astype(mx.int32)).astype(mx.int32)
    count = int(cast('int', mx.sum(mask).tolist()))
    rows = mx.array([], dtype=mx.int32) if count == 0 else ordering[-count:]
    return prune(x, rows)


def topk_rows(
    x: SparseTensor,
    counts: Sequence[int],
    *,
    rho: float = 1.0,
) -> mx.array:
    if rho <= 0:
        raise ValueError('rho must be positive.')

    row_counts = x.batch_counts
    if row_counts is None:
        raise ValueError('batch_counts metadata is required for topk_rows.')
    if len(counts) != len(row_counts):
        raise ValueError('counts must match the batch count.')

    selected = []
    start = 0
    for keep, row_count in zip(counts, row_counts, strict=True):
        stop = start + int(row_count)
        if stop > x.capacity:
            raise ValueError(
                'batch row counts exceed sparse tensor row count.'
            )
        rows = mx.arange(start, stop, dtype=mx.int32)
        start = stop

        k = min(int(int(keep) * rho), int(rows.shape[0]))
        if k <= 0:
            continue
        scores = mx.take(x.feats[:, 0], rows, axis=0)
        order = mx.argsort(scores)
        selected.append(mx.take(rows, order[-k:], axis=0))

    if start != x.capacity:
        raise ValueError('counts must cover all sparse tensor rows.')
    if not selected:
        return mx.array([], dtype=mx.int32)
    return mx.concatenate(selected, axis=0)
