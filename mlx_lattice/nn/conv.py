from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import KernelSpec, SparseTensor
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

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return conv3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
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
