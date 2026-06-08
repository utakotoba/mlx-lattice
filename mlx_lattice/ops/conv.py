from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core import KernelSpec, SparseTensor
from mlx_lattice.core.types import Triple

__all__ = [
    'conv3d',
    'conv_transpose3d',
    'generative_conv_transpose3d',
    'subm_conv3d',
]


def conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if spec.is_pointwise:
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    return _fused_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='forward',
        output_stride=_mul_stride(x.stride, spec.stride),
    )


def subm_conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=1,
        padding=0,
        dilation=dilation,
    )
    _require_odd_kernel(spec.size, 'subm_conv3d')
    if spec.size == (1, 1, 1) and spec.dilation == (1, 1, 1):
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    return _fused_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='forward',
        output_stride=x.stride,
        reuse_input_coords=True,
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
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return _fused_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='transposed',
        output_stride=_div_stride(x.stride, spec.stride),
    )


def generative_conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    spec = KernelSpec(size=kernel_size, stride=stride)
    return _fused_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='generative',
        output_stride=_div_stride(x.stride, spec.stride),
    )


# MARK: - execution policy


def _fused_conv(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None,
    spec: KernelSpec,
    *,
    map_kind: str,
    output_stride: Triple,
    reuse_input_coords: bool = False,
) -> SparseTensor:
    _validate_feature_dtype(x.feats, weight)
    mapped_weight = _mapped_weight_for_kernel(x, weight, spec.volume)
    out_coords, feats, counts = ext.sparse_conv(
        x.coords,
        x.active_rows,
        x.feats,
        mapped_weight,
        map_kind,
        list(spec.size),
        list(spec.stride),
        list(spec.padding),
        list(spec.dilation),
    )
    if reuse_input_coords:
        return x.replace(feats=_with_bias(feats, bias))
    return SparseTensor(
        out_coords,
        _with_bias(feats, bias),
        stride=output_stride,
        coord_manager=x.coord_manager,
        active_rows=counts[1:2],
    )


def _pointwise_features(x: SparseTensor, weight: mx.array) -> mx.array:
    _validate_feature_dtype(x.feats, weight)
    matrix = _pointwise_weight_matrix(x, weight)
    return x.feats @ matrix.T


# MARK: - validation


def _validate_feature_dtype(feats: mx.array, weight: mx.array) -> None:
    if feats.dtype != mx.float32 or weight.dtype != mx.float32:
        raise ValueError('convolution currently supports float32 tensors.')


def _pointwise_weight_matrix(x: SparseTensor, weight: mx.array) -> mx.array:
    if weight.ndim == 2:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight
    if (
        weight.ndim == 5
        and weight.shape[1] == 1
        and weight.shape[2] == 1
        and weight.shape[3] == 1
    ):
        if weight.shape[4] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[:, 0, 0, 0, :]
    if weight.ndim == 3 and weight.shape[0] == 1:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[0].T
    raise ValueError(
        'pointwise weight must have shape (C_out, C_in), '
        '(C_out, 1, 1, 1, C_in), or (1, C_in, C_out).'
    )


def _mapped_weight_for_kernel(
    x: SparseTensor,
    weight: mx.array,
    kernel_rows: int,
) -> mx.array:
    if weight.ndim == 3:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        if weight.shape[0] != kernel_rows:
            raise ValueError(
                'weight kernel rows must match the convolution kernel.'
            )
        return weight

    if weight.ndim != 5:
        raise ValueError(
            'mapped convolution weight must have shape (K, C_in, C_out) '
            'or (C_out, Kx, Ky, Kz, C_in).'
        )
    if weight.shape[4] != x.channels:
        raise ValueError('weight input channels must match x.channels.')
    if (
        int(weight.shape[1] * weight.shape[2] * weight.shape[3])
        != kernel_rows
    ):
        raise ValueError(
            'weight kernel rows must match the convolution kernel.'
        )
    out_channels = int(weight.shape[0])
    return weight.reshape(out_channels, kernel_rows, x.channels).transpose(
        1, 2, 0
    )


def _with_bias(feats: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return feats
    if bias.ndim != 1:
        raise ValueError('bias must have shape (C_out,).')
    if bias.shape[0] != feats.shape[1]:
        raise ValueError('bias channels must match output channels.')
    if bias.dtype != feats.dtype:
        raise ValueError('bias dtype must match output features.')
    return feats + bias


def _require_odd_kernel(values: Triple, op_name: str) -> None:
    if any(value % 2 == 0 for value in values):
        raise ValueError(f'{op_name} requires odd kernel_size values.')


def _mul_stride(lhs: Triple, rhs: Triple) -> Triple:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])


def _div_stride(lhs: Triple, rhs: Triple) -> Triple:
    out = []
    for left, right in zip(lhs, rhs, strict=True):
        if left % right != 0:
            raise ValueError(
                'transpose stride must divide the input tensor stride.'
            )
        out.append(left // right)
    return (out[0], out[1], out[2])
