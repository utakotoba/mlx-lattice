from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import mlx.core as mx

from mlx_lattice.core.coords import SparseAlignment, build_sparse_alignment
from mlx_lattice.core.tensor import SparseTensor
from mlx_lattice.core.types import triple

type SparseJoin = Literal['inner', 'left', 'right', 'outer']


def sparse_collate(
    coords: Sequence[mx.array],
    feats: Sequence[mx.array],
    *,
    stride: int | Sequence[int] = 1,
) -> SparseTensor:
    """Collate unbatched sparse coordinates/features into one tensor.

    Input coordinates must have shape ``(N, 3)``; the function prepends a
    batch column and records ``batch_counts`` metadata. Feature arrays must
    each have shape ``(N_i, C)`` and all coordinate arrays must share dtype.
    """
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


def align_sparse(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'inner',
) -> SparseAlignment:
    """Build coordinate-value alignment metadata for two sparse tensors.

    Both tensors must have the same sparse stride and coordinate dtype. The
    returned row maps can be used to gather feature rows under ``inner``,
    ``left``, ``right``, or ``outer`` sparse support semantics.
    """
    _require_compatible_sparse_tensors(lhs, rhs)
    return build_sparse_alignment(
        lhs.coords,
        lhs.active_rows,
        rhs.coords,
        rhs.active_rows,
        join=join,
    )


def gather_aligned_features(
    x: SparseTensor,
    rows: mx.array,
    *,
    fill: float = 0.0,
) -> mx.array:
    """Gather sparse features with ``-1`` rows filled by ``fill``.

    ``rows`` must be an ``int32`` vector. Non-negative entries gather
    ``x.feats``; negative entries produce the scalar fill value in every
    channel.
    """
    if rows.ndim != 1 or rows.dtype != mx.int32:
        raise ValueError('rows must have shape (N,) and int32 dtype.')
    clipped = mx.maximum(rows, 0)
    gathered = mx.take(x.feats, clipped, axis=0)
    valid = (rows >= 0).astype(x.feats.dtype)[:, None]
    if fill == 0.0:
        return gathered * valid
    fill_value = mx.array(float(fill), dtype=x.feats.dtype)
    return mx.where(valid.astype(mx.bool_), gathered, fill_value)


def sparse_binary_op(
    lhs: SparseTensor,
    rhs: SparseTensor,
    op: Literal['add', 'sub', 'mul', 'maximum', 'minimum'],
    *,
    join: SparseJoin = 'outer',
    lhs_fill: float = 0.0,
    rhs_fill: float = 0.0,
) -> SparseTensor:
    """Apply an elementwise binary op after coordinate-value alignment.

    If operands share coordinate identity, the operation runs directly on
    feature matrices. Otherwise it builds a sparse alignment and fills missing
    rows according to ``lhs_fill`` and ``rhs_fill``.
    """
    _require_compatible_sparse_tensors(lhs, rhs)
    if lhs.channels != rhs.channels:
        raise ValueError(
            'sparse binary operands must have matching channels.'
        )
    if lhs.same_coords(rhs):
        return lhs.replace(feats=_apply_binary_op(lhs.feats, rhs.feats, op))
    alignment = align_sparse(lhs, rhs, join=join)
    lhs_feats = gather_aligned_features(
        lhs, alignment.lhs_rows, fill=lhs_fill
    )
    rhs_feats = gather_aligned_features(
        rhs, alignment.rhs_rows, fill=rhs_fill
    )
    return SparseTensor(
        alignment.coords,
        _apply_binary_op(lhs_feats, rhs_feats, op),
        lhs.stride,
        coord_manager=lhs.coord_manager,
        active_rows=alignment.active_rows,
    )


def sparse_add(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'outer',
) -> SparseTensor:
    """Add sparse tensors after coordinate alignment."""
    return sparse_binary_op(lhs, rhs, 'add', join=join)


def sparse_sub(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'outer',
) -> SparseTensor:
    """Subtract sparse tensors after coordinate alignment."""
    return sparse_binary_op(lhs, rhs, 'sub', join=join)


def sparse_mul(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'inner',
) -> SparseTensor:
    """Multiply sparse tensors after coordinate alignment."""
    return sparse_binary_op(lhs, rhs, 'mul', join=join)


def sparse_maximum(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'inner',
) -> SparseTensor:
    """Take elementwise maximum after coordinate alignment."""
    return sparse_binary_op(lhs, rhs, 'maximum', join=join)


def sparse_minimum(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'inner',
) -> SparseTensor:
    """Take elementwise minimum after coordinate alignment."""
    return sparse_binary_op(lhs, rhs, 'minimum', join=join)


def cat(
    tensors: Sequence[SparseTensor],
    *,
    join: SparseJoin = 'inner',
) -> SparseTensor:
    """Concatenate sparse features, aligning coordinates when needed.

    Multiple tensors are supported when they already share coordinate identity.
    Value-aligned concatenation currently accepts two tensors because the join
    support and fill behavior must be unambiguous.
    """
    if not tensors:
        raise ValueError('expected at least one sparse tensor.')

    first = tensors[0]
    if all(first.same_coords(tensor) for tensor in tensors[1:]):
        return first.replace(
            feats=mx.concatenate(
                [tensor.feats for tensor in tensors], axis=1
            )
        )
    if len(tensors) != 2:
        raise ValueError(
            'value-aligned cat currently supports exactly two sparse tensors.'
        )
    return sparse_cat_aligned(first, tensors[1], join=join)


def sparse_cat_aligned(
    lhs: SparseTensor,
    rhs: SparseTensor,
    *,
    join: SparseJoin = 'inner',
) -> SparseTensor:
    """Concatenate two sparse tensors after coordinate-value alignment."""
    _require_compatible_sparse_tensors(lhs, rhs)
    if lhs.same_coords(rhs):
        return lhs.replace(
            feats=mx.concatenate([lhs.feats, rhs.feats], axis=1)
        )
    alignment = align_sparse(lhs, rhs, join=join)
    lhs_feats = gather_aligned_features(lhs, alignment.lhs_rows)
    rhs_feats = gather_aligned_features(rhs, alignment.rhs_rows)
    return SparseTensor(
        alignment.coords,
        mx.concatenate([lhs_feats, rhs_feats], axis=1),
        lhs.stride,
        coord_manager=lhs.coord_manager,
        active_rows=alignment.active_rows,
    )


def crop(
    x: SparseTensor,
    *,
    min_coord: Sequence[int],
    max_coord: Sequence[int],
) -> SparseTensor:
    """Crop a sparse tensor to an inclusive spatial coordinate box.

    ``min_coord`` and ``max_coord`` are spatial triples over ``x, y, z``. The
    batch column is not filtered.
    """
    lower = _spatial_bound(min_coord, 'min_coord')
    upper = _spatial_bound(max_coord, 'max_coord')
    if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError('min_coord must be <= max_coord.')
    spatial = x.coords[:, 1:]
    active = mx.arange(x.capacity, dtype=mx.int32) < x.active_rows[0]
    mask = active
    for axis, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        values = spatial[:, axis]
        mask = mask & (values >= lo) & (values <= hi)
    return prune_mask(x, mask)


def replace_feature(x: SparseTensor, feats: mx.array) -> SparseTensor:
    """Return ``x`` with its feature matrix replaced.

    The replacement must satisfy the ``SparseTensor`` row contract: same row
    count as ``x.coords`` and two-dimensional ``(N, C_new)`` shape.
    """
    return x.replace(feats=feats)


def _apply_binary_op(lhs: mx.array, rhs: mx.array, op: str) -> mx.array:
    if op == 'add':
        return lhs + rhs
    if op == 'sub':
        return lhs - rhs
    if op == 'mul':
        return lhs * rhs
    if op == 'maximum':
        return mx.maximum(lhs, rhs)
    if op == 'minimum':
        return mx.minimum(lhs, rhs)
    raise ValueError(f'unknown sparse binary op: {op}.')


def _require_compatible_sparse_tensors(
    lhs: SparseTensor,
    rhs: SparseTensor,
) -> None:
    if lhs.stride != rhs.stride:
        raise ValueError('sparse tensor strides must match.')
    if lhs.coords.dtype != rhs.coords.dtype:
        raise ValueError('sparse tensor coordinate dtypes must match.')


def _spatial_bound(value: Sequence[int], name: str) -> tuple[int, int, int]:
    raw = tuple(int(item) for item in value)
    if len(raw) != 3:
        raise ValueError(f'{name} must contain exactly 3 values.')
    return (raw[0], raw[1], raw[2])


def prune(x: SparseTensor, rows: mx.array) -> SparseTensor:
    """Keep the specified sparse rows.

    ``rows`` is a one-dimensional index vector. The result contains gathered
    coordinates and features and receives a new coordinate key in the same
    manager.
    """
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
    """Keep sparse rows selected by a boolean mask.

    ``mask`` must have length equal to the sparse buffer capacity. Selected
    rows are gathered into a compact sparse tensor.
    """
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
    """Select top-scoring rows per batch using the first feature channel.

    ``counts`` gives the requested rows per batch and ``rho`` scales those
    counts before clamping to the available rows in each batch. Requires
    ``x.batch_counts``.
    """
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
