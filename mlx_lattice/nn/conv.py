from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import (
    CoordinateMapKey,
    KernelSpec,
    SparseTensor,
)
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)

__all__ = [
    'Conv3d',
    'ConvTranspose3d',
    'GenerativeConvTranspose3d',
    'SubmConv3d',
]

if TYPE_CHECKING:
    from mlx_lattice.nn.quantized_conv import (
        QuantizedConv3d,
        QuantizedConvTranspose3d,
        QuantizedGenerativeConvTranspose3d,
        QuantizedSubmConv3d,
    )


class Conv3d(mxnn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        _init_kernel_params(
            self, in_channels, out_channels, self.spec, bias
        )

    def __call__(
        self,
        x: SparseTensor,
        *,
        coordinates: SparseTensor
        | CoordinateMapKey
        | mx.array
        | None = None,
    ) -> SparseTensor:
        return conv3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
            coordinates=coordinates,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        *,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedConv3d:
        from mlx_lattice.nn.quantized_conv import QuantizedConv3d

        _validate_quantize_request(mode, quantize_input)
        return QuantizedConv3d.from_conv(
            self, group_size=group_size, bits=4 if bits is None else bits
        )


class SubmConv3d(mxnn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec = KernelSpec(
            size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        )
        _init_kernel_params(
            self, in_channels, out_channels, self.spec, bias
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return subm_conv3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            dilation=self.spec.dilation,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        *,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedSubmConv3d:
        from mlx_lattice.nn.quantized_conv import QuantizedSubmConv3d

        _validate_quantize_request(mode, quantize_input)
        return QuantizedSubmConv3d.from_conv(
            self, group_size=group_size, bits=4 if bits is None else bits
        )


class ConvTranspose3d(mxnn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        _init_kernel_params(
            self, in_channels, out_channels, self.spec, bias
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return conv_transpose3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        *,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedConvTranspose3d:
        from mlx_lattice.nn.quantized_conv import QuantizedConvTranspose3d

        _validate_quantize_request(mode, quantize_input)
        return QuantizedConvTranspose3d.from_conv(
            self, group_size=group_size, bits=4 if bits is None else bits
        )


class GenerativeConvTranspose3d(mxnn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec = KernelSpec(size=kernel_size, stride=stride)
        _init_kernel_params(
            self, in_channels, out_channels, self.spec, bias
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return generative_conv_transpose3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        *,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedGenerativeConvTranspose3d:
        from mlx_lattice.nn.quantized_conv import (
            QuantizedGenerativeConvTranspose3d,
        )

        _validate_quantize_request(mode, quantize_input)
        return QuantizedGenerativeConvTranspose3d.from_conv(
            self, group_size=group_size, bits=4 if bits is None else bits
        )


def _init_kernel_params(
    module: mxnn.Module,
    in_channels: int,
    out_channels: int,
    spec: KernelSpec,
    bias: bool,
) -> None:
    scale = math.sqrt(1.0 / (in_channels * spec.volume))
    module.weight = mx.random.uniform(
        low=-scale,
        high=scale,
        shape=(out_channels, *spec.size, in_channels),
    )
    if bias:
        module.bias = mx.zeros((out_channels,))


def _optional_bias(module: mxnn.Module) -> mx.array | None:
    return module.bias if 'bias' in module else None


def _validate_quantize_request(mode: str, quantize_input: bool) -> None:
    if mode != 'affine':
        raise ValueError(
            'quantized sparse convolution supports affine mode.'
        )
    if quantize_input:
        raise ValueError(
            'quantized sparse convolution uses floating-point activations.'
        )
