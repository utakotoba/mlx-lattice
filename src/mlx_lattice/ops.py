from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mlx.core as mx

from mlx_lattice._native import conv3d_feats as _conv3d_feats
from mlx_lattice._native import (
    conv3d_residual_feats as _conv3d_residual_feats,
)
from mlx_lattice.point import (
    build_generative_map,
    downsample,
    kernel_offsets,
)
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import triple


def sparse_collate(
    coords: Sequence[mx.array],
    feats: Sequence[mx.array],
) -> SparseTensor:
    if len(coords) != len(feats):
        raise ValueError('coords and feats batch sizes must match.')

    batched_coords = []
    for batch, values in enumerate(coords):
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError('collated coords must have shape (N, 3).')
        batch_col = mx.full((values.shape[0], 1), batch, dtype=values.dtype)
        batched_coords.append(mx.concatenate([batch_col, values], axis=1))
    return SparseTensor(
        mx.concatenate(batched_coords, axis=0),
        mx.concatenate(list(feats), axis=0),
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
    )


def topk_rows(
    x: SparseTensor,
    counts: Sequence[int],
    *,
    rho: float = 1.0,
) -> mx.array:
    if rho <= 0:
        raise ValueError('rho must be positive.')

    selected = []
    start = 0
    row_counts = x.batch_counts
    if row_counts is None:
        row_counts = tuple(int(rows.shape[0]) for rows in x.batch_rows)
    if len(counts) != len(row_counts):
        raise ValueError('counts must match the batch count.')

    for keep, row_count in zip(counts, row_counts, strict=True):
        stop = start + int(row_count)
        if stop > x.n_points:
            raise ValueError(
                'batch row counts exceed sparse tensor row count.'
            )
        rows = mx.arange(start, stop, dtype=mx.int32)
        start = stop
        k = min(int(keep * rho), int(rows.shape[0]))
        if k <= 0:
            continue
        scores = mx.take(x.feats[:, 0], rows, axis=0)
        order = mx.argsort(scores)
        selected.append(mx.take(rows, order[-k:], axis=0))
    if start != x.n_points:
        raise ValueError('counts must cover all sparse tensor rows.')
    if not selected:
        return mx.array([], dtype=mx.int32)
    return mx.concatenate(selected, axis=0)


def linear(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
) -> SparseTensor:
    if weight.ndim != 2:
        raise ValueError('weight must have shape (Cout, Cin).')
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )
    feats = x.feats @ mx.swapaxes(weight, 0, 1)
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError('bias must have shape (Cout,).')
        feats = feats + bias
    return x.replace(feats=feats)


def relu(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.maximum(x.feats, 0))


def sigmoid(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.sigmoid(x.feats))


def conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    transposed: bool = False,
    weight_layout: Literal['flat', 'mlx'] | None = None,
) -> SparseTensor:
    if transposed:
        raise NotImplementedError(
            'transposed sparse conv is not implemented.'
        )
    if triple(padding, name='padding') != (0, 0, 0):
        raise NotImplementedError(
            'sparse conv currently supports padding=0.'
        )
    if triple(dilation, name='dilation') != (1, 1, 1):
        raise NotImplementedError(
            'sparse conv currently supports dilation=1.'
        )
    kernel = triple(kernel_size, name='kernel_size')
    weight = _normalize_weight(weight, kernel, weight_layout)
    if weight.dtype != mx.float32 or x.feats.dtype != mx.float32:
        raise ValueError('conv3d currently supports float32 tensors.')
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )

    op_stride = triple(stride, name='stride')
    if op_stride == (1, 1, 1) and kernel == (1, 1, 1):
        feats = x.feats @ weight[0]
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[2]:
                raise ValueError('bias must have shape (Cout,).')
            feats = feats + bias
        return x.replace(feats=feats)

    mapping = x.kernel_map(kernel_size=kernel, stride=op_stride)
    if weight.shape[0] != len(mapping.offsets):
        raise ValueError(
            'weight kernel dimension does not match kernel_size.'
        )

    out_rows = int(mapping.out_coords.shape[0])
    if (
        op_stride == (1, 1, 1)
        and out_rows == x.n_points
        and mapping.center >= 0
    ):
        center = x.feats @ weight[mapping.center]
        feats = _conv3d_residual_feats(
            center,
            x.feats,
            weight,
            mapping.residual_maps,
            mapping.residual_kernels,
            mapping.residual_offsets,
        )
    else:
        feats = _conv3d_feats(
            x.feats,
            weight,
            mapping.maps,
            mapping.kernels,
            out_rows,
        )
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[2]:
            raise ValueError('bias must have shape (Cout,).')
        feats = feats + bias

    out_stride = tuple(
        a * b for a, b in zip(x.stride, op_stride, strict=True)
    )
    return SparseTensor(mapping.out_coords, feats, out_stride)


def generative_conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    weight_layout: Literal['flat', 'mlx'] | None = None,
) -> SparseTensor:
    kernel = triple(kernel_size, name='kernel_size')
    op_stride = triple(stride, name='stride')
    weight = _normalize_weight(weight, kernel, weight_layout)
    if weight.dtype != mx.float32 or x.feats.dtype != mx.float32:
        raise ValueError(
            'transpose conv currently supports float32 tensors.'
        )
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )
    if any(a % b != 0 for a, b in zip(x.stride, op_stride, strict=True)):
        raise ValueError('input tensor stride must be divisible by stride.')

    mapping = build_generative_map(
        x.coords, kernel_size=kernel, stride=op_stride
    )
    feats = _conv3d_feats(
        x.feats,
        weight,
        mapping.maps,
        mapping.kernels,
        int(mapping.out_coords.shape[0]),
    )
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[2]:
            raise ValueError('bias must have shape (Cout,).')
        feats = feats + bias

    out_stride = tuple(
        a // b for a, b in zip(x.stride, op_stride, strict=True)
    )
    return SparseTensor(mapping.out_coords, feats, out_stride)


def pool3d(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    volume = len(kernel_offsets(kernel_size))
    weight = mx.ones((volume, x.channels, x.channels), dtype=x.feats.dtype)
    return conv3d(x, weight, kernel_size=kernel_size, stride=stride)


spdownsample = downsample
sparse_conv3d = conv3d
generative_sparse_conv_transpose3d = generative_conv_transpose3d
sparse_pool3d = pool3d


def _normalize_weight(
    weight: mx.array,
    kernel_size: tuple[int, int, int],
    layout: Literal['flat', 'mlx'] | None,
) -> mx.array:
    if layout is None:
        layout = 'mlx' if weight.ndim == 5 else 'flat'

    if layout == 'flat':
        if weight.ndim != 3:
            raise ValueError('flat weight must have shape (K, Cin, Cout).')
        return weight

    if layout != 'mlx':
        raise ValueError("weight_layout must be 'flat', 'mlx', or None.")
    if weight.ndim != 5:
        raise ValueError(
            'MLX weight must have shape (Cout, Kx, Ky, Kz, Cin).'
        )
    if tuple(int(v) for v in weight.shape[1:4]) != kernel_size:
        raise ValueError('MLX weight spatial shape must match kernel_size.')

    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[4])
    return mx.moveaxis(weight, 0, -1).reshape(
        (-1, in_channels, out_channels)
    )
