from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mlx.core as mx

from mlx_lattice.core import KernelSpec, SparseTensor
from mlx_lattice.core.types import Triple
from mlx_lattice.ops._relation_exec import (
    sparse_pool_features_from_relation,
)

PoolMode = Literal['sum', 'max', 'avg']

__all__ = [
    'avg_pool3d',
    'global_avg_pool',
    'global_max_pool',
    'global_sum_pool',
    'max_pool3d',
    'pool3d',
    'sum_pool3d',
]


def pool3d(
    x: SparseTensor,
    *,
    mode: PoolMode = 'sum',
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    """Apply local sparse 3D pooling with ``sum``, ``max``, or ``avg`` mode.

    Local pooling builds a forward kernel relation and reduces input features
    that contribute to each output coordinate. The output sparse stride is
    ``x.stride * stride``. Current native pooling routes accept ``float32``
    features; Metal routes additionally require ``int32`` coordinates.
    """
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return _fused_pool(
        x,
        spec,
        mode,
        output_stride=_mul_stride(x.stride, spec.stride),
    )


def sum_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    """Apply local sparse sum pooling.

    The result feature at each output row is the sum of all contributing input
    rows in the sparse kernel relation.
    """
    return pool3d(
        x,
        mode='sum',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def max_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    """Apply local sparse max pooling.

    The result feature at each output row is the channel-wise maximum over
    contributing input rows in the sparse kernel relation.
    """
    return pool3d(
        x,
        mode='max',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def avg_pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    """Apply local sparse average pooling.

    The result feature at each output row is the sparse sum divided by the
    number of contributing relation edges for that output row.
    """
    return pool3d(
        x,
        mode='avg',
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def global_sum_pool(x: SparseTensor) -> mx.array:
    """Sum features independently for each batch.

    Requires ``x.batch_counts`` and returns a dense ``(B, C)`` MLX array.
    """
    return _stack_batch_reductions(x, mode='sum')


def global_avg_pool(x: SparseTensor) -> mx.array:
    """Average features independently for each batch.

    Requires ``x.batch_counts`` and returns a dense ``(B, C)`` MLX array.
    Empty batches produce zero rows.
    """
    return _stack_batch_reductions(x, mode='avg')


def global_max_pool(x: SparseTensor) -> mx.array:
    """Max-reduce features independently for each batch.

    Requires ``x.batch_counts`` and returns a dense ``(B, C)`` MLX array.
    Empty batches are rejected because max has no neutral finite sparse row.
    """
    return _stack_batch_reductions(x, mode='max')


# MARK: - local pooling


def _fused_pool(
    x: SparseTensor,
    spec: KernelSpec,
    mode: PoolMode,
    *,
    output_stride: Triple,
) -> SparseTensor:
    _validate_pool_dtype(x.feats)
    _validate_metal_coord_dtype(x)
    if mode not in ('sum', 'max', 'avg'):
        raise ValueError("mode must be 'sum', 'max', or 'avg'.")
    relation = x.coord_manager.kernel_relation(
        x.coord_key,
        kernel_size=spec.size,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
    )
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    if relation.out_coords is None:
        raise ValueError('kernel relation is missing output coordinates.')
    feats = sparse_pool_features_from_relation(
        x.feats,
        relation,
        input_exclusive=_input_exclusive(spec),
        mode=mode,
    )
    return SparseTensor(
        relation.out_coords,
        feats,
        stride=output_stride,
        coord_manager=x.coord_manager,
        active_rows=relation.out_count,
    )


# MARK: - global pooling


def _stack_batch_reductions(x: SparseTensor, *, mode: PoolMode) -> mx.array:
    _validate_pool_dtype(x.feats)
    counts = _require_batch_counts(x)
    if len(counts) == 1:
        if mode == 'sum':
            return mx.sum(x.feats, axis=0, keepdims=True)
        if mode == 'avg':
            denom = mx.array(max(counts[0], 1), dtype=x.feats.dtype)
            return mx.sum(x.feats, axis=0, keepdims=True) / denom
        if mode == 'max':
            if counts[0] == 0:
                raise ValueError(
                    'global_max_pool does not support empty batches.'
                )
            return mx.max(x.feats, axis=0, keepdims=True)

    batch_ids = _batch_ids(counts)
    shape = (len(counts), x.channels)

    if mode == 'sum':
        return (
            mx.zeros(shape, dtype=x.feats.dtype).at[batch_ids].add(x.feats)
        )
    if mode == 'avg':
        sums = (
            mx.zeros(shape, dtype=x.feats.dtype).at[batch_ids].add(x.feats)
        )
        denom = mx.array(counts, dtype=x.feats.dtype).reshape((-1, 1))
        return mx.where(denom > 0, sums / mx.maximum(denom, 1), sums)
    if mode == 'max':
        if any(count == 0 for count in counts):
            raise ValueError(
                'global_max_pool does not support empty batches.'
            )
        return (
            mx.full(shape, -float('inf'), dtype=x.feats.dtype)
            .at[batch_ids]
            .maximum(x.feats)
        )
    raise ValueError("mode must be 'sum', 'max', or 'avg'.")


def _require_batch_counts(x: SparseTensor) -> tuple[int, ...]:
    if x.batch_counts is None:
        raise ValueError(
            'batch_counts metadata is required for global pooling.'
        )
    return x.batch_counts


def _batch_ids(counts: tuple[int, ...]) -> mx.array:
    ids = [
        mx.full((count,), batch, dtype=mx.int32)
        for batch, count in enumerate(counts)
        if count > 0
    ]
    if not ids:
        return mx.array([], dtype=mx.int32)
    return mx.concatenate(ids, axis=0)


# MARK: - validation


def _validate_pool_dtype(feats: mx.array) -> None:
    if feats.dtype != mx.float32:
        raise ValueError('pooling currently supports float32 tensors.')


def _validate_metal_coord_dtype(x: SparseTensor) -> None:
    if mx.default_device() == mx.gpu and x.coords.dtype != mx.int32:
        raise ValueError('Metal sparse pooling requires int32 coordinates.')


def _input_exclusive(spec: KernelSpec) -> bool:
    return all(
        stride >= (size - 1) * dilation + 1
        for stride, size, dilation in zip(
            spec.stride,
            spec.size,
            spec.dilation,
            strict=True,
        )
    )


def _mul_stride(lhs: Triple, rhs: Triple) -> Triple:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])
