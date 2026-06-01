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
    build_transposed_kernel_map,
)
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import triple


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
        return conv_transpose3d(
            x,
            weight,
            bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            weight_layout=weight_layout,
        )
    kernel = triple(kernel_size, name='kernel_size')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    if any(value < 0 for value in pad):
        raise ValueError('padding values must be non-negative.')
    if any(value <= 0 for value in rate):
        raise ValueError('dilation values must be positive.')
    weight = _normalize_weight(weight, kernel, weight_layout)
    if weight.dtype != mx.float32 or x.feats.dtype != mx.float32:
        raise ValueError('conv3d currently supports float32 tensors.')
    if weight.shape[1] != x.channels:
        raise ValueError(
            'weight input channels must match tensor features.'
        )

    op_stride = triple(stride, name='stride')
    if (
        op_stride == (1, 1, 1)
        and kernel == (1, 1, 1)
        and pad == (0, 0, 0)
        and rate == (1, 1, 1)
    ):
        feats = x.feats @ weight[0]
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[2]:
                raise ValueError('bias must have shape (Cout,).')
            feats = feats + bias
        return x.replace(feats=feats)

    mapping = x.kernel_map(
        kernel_size=kernel,
        stride=op_stride,
        padding=pad,
        dilation=rate,
    )
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

    if op_stride == (1, 1, 1) and out_rows == x.n_points:
        return x.replace(feats=feats)

    out_stride = tuple(
        a * b for a, b in zip(x.stride, op_stride, strict=True)
    )
    return SparseTensor(
        mapping.out_coords,
        feats,
        out_stride,
        coord_manager=x.coord_manager,
    )


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
    return SparseTensor(
        mapping.out_coords,
        feats,
        out_stride,
        coord_manager=x.coord_manager,
    )


def conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    weight_layout: Literal['flat', 'mlx'] | None = None,
) -> SparseTensor:
    kernel = triple(kernel_size, name='kernel_size')
    op_stride = triple(stride, name='stride')
    pad = triple(padding, name='padding')
    rate = triple(dilation, name='dilation')
    if any(value < 0 for value in pad):
        raise ValueError('padding values must be non-negative.')
    if any(value <= 0 for value in rate):
        raise ValueError('dilation values must be positive.')

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

    mapping = build_transposed_kernel_map(
        x.coords,
        kernel_size=kernel,
        stride=op_stride,
        padding=pad,
        dilation=rate,
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
    return SparseTensor(
        mapping.out_coords,
        feats,
        out_stride,
        coord_manager=x.coord_manager,
    )


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
