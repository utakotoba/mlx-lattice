from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    linear,
    pool3d,
    relu,
    sigmoid,
)
from mlx_lattice.tensor import SparseTensor
from mlx_lattice.types import Triple, triple


class Linear(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('channels must be positive.')
        scale = math.sqrt(1 / in_channels)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: SparseTensor) -> SparseTensor:
        bias = self.bias if 'bias' in self else None
        return linear(x, self.weight, bias)


class ReLU(nn.Module):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return relu(x)


class Sigmoid(nn.Module):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return sigmoid(x)


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm(num_features, eps, momentum, affine)

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=self.norm(x.feats))


class Conv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('channels must be positive.')

        kernel = triple(kernel_size, name='kernel_size')
        if any(size <= 0 for size in kernel):
            raise ValueError('kernel_size values must be positive.')

        scale = math.sqrt(1 / (in_channels * _volume(kernel)))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.kernel_size = kernel
        self.stride = triple(stride, name='stride')
        self.padding = triple(padding, name='padding')
        self.dilation = triple(dilation, name='dilation')

    def __call__(self, x: SparseTensor) -> SparseTensor:
        bias = self.bias if 'bias' in self else None
        return conv3d(
            x,
            self.weight,
            bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            weight_layout='mlx',
        )


class GenerativeConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('channels must be positive.')
        kernel = triple(kernel_size, name='kernel_size')
        if any(size <= 0 for size in kernel):
            raise ValueError('kernel_size values must be positive.')

        scale = math.sqrt(1 / (in_channels * _volume(kernel)))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.kernel_size = kernel
        self.stride = triple(stride, name='stride')

    def __call__(self, x: SparseTensor) -> SparseTensor:
        bias = self.bias if 'bias' in self else None
        return generative_conv_transpose3d(
            x,
            self.weight,
            bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            weight_layout='mlx',
        )


class ConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('channels must be positive.')
        kernel = triple(kernel_size, name='kernel_size')
        if any(size <= 0 for size in kernel):
            raise ValueError('kernel_size values must be positive.')

        scale = math.sqrt(1 / (in_channels * _volume(kernel)))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.kernel_size = kernel
        self.stride = triple(stride, name='stride')
        self.padding = triple(padding, name='padding')
        self.dilation = triple(dilation, name='dilation')

    def __call__(self, x: SparseTensor) -> SparseTensor:
        bias = self.bias if 'bias' in self else None
        return conv_transpose3d(
            x,
            self.weight,
            bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            weight_layout='mlx',
        )


class SumPool3d(nn.Module):
    def __init__(
        self,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = triple(kernel_size, name='kernel_size')
        self.stride = (
            self.kernel_size
            if stride is None
            else triple(stride, name='stride')
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return pool3d(x, kernel_size=self.kernel_size, stride=self.stride)


Pool3d = SumPool3d
SparseConv3d = Conv3d
SparseConvTranspose3d = ConvTranspose3d
SparseGenerativeConvTranspose3d = GenerativeConvTranspose3d
SparseLinear = Linear


def _volume(values: Triple) -> int:
    return values[0] * values[1] * values[2]
