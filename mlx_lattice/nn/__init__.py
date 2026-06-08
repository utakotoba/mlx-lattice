from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import KernelSpec, SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    pool3d,
    subm_conv3d,
    sum_pool3d,
)

PoolMode = Literal['sum', 'max', 'avg']

__all__ = [
    'GELU',
    'AvgPool3d',
    'BatchNorm',
    'Conv3d',
    'ConvTranspose3d',
    'Dropout',
    'GenerativeConvTranspose3d',
    'GlobalAvgPool',
    'GlobalMaxPool',
    'GlobalSumPool',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'MaxPool3d',
    'Pool3d',
    'RMSNorm',
    'ReLU',
    'SiLU',
    'Sigmoid',
    'Softplus',
    'SubmConv3d',
    'SumPool3d',
    'Tanh',
]


# MARK: - feature modules


class Linear(mxnn.Linear):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class ReLU(mxnn.ReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Sigmoid(mxnn.Sigmoid):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class GELU(mxnn.GELU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class SiLU(mxnn.SiLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class LeakyReLU(mxnn.LeakyReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Tanh(mxnn.Tanh):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Softplus(mxnn.Softplus):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Dropout(mxnn.Dropout):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class BatchNorm(mxnn.BatchNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class LayerNorm(mxnn.LayerNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class RMSNorm(mxnn.RMSNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


# MARK: - convolution modules


class Conv3d(mxnn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
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
            self, input_dims, output_dims, self.spec.volume, bias
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
        input_dims: int,
        output_dims: int,
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
            self, input_dims, output_dims, self.spec.volume, bias
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
        input_dims: int,
        output_dims: int,
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
            self, input_dims, output_dims, self.spec.volume, bias
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
        input_dims: int,
        output_dims: int,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spec = KernelSpec(size=kernel_size, stride=stride)
        _init_kernel_params(
            self, input_dims, output_dims, self.spec.volume, bias
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return generative_conv_transpose3d(
            x,
            self.weight,
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
        )


# MARK: - pooling modules


class Pool3d(mxnn.Module):
    def __init__(
        self,
        *,
        mode: PoolMode = 'sum',
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return pool3d(
            x,
            mode=self.mode,
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )


class SumPool3d(Pool3d):
    def __init__(
        self,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        super().__init__(
            mode='sum',
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return sum_pool3d(
            x,
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )


class MaxPool3d(Pool3d):
    def __init__(
        self,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        super().__init__(
            mode='max',
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return max_pool3d(
            x,
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )


class AvgPool3d(Pool3d):
    def __init__(
        self,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        super().__init__(
            mode='avg',
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return avg_pool3d(
            x,
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )


class GlobalSumPool(mxnn.Module):
    def __call__(self, x: SparseTensor) -> mx.array:
        return global_sum_pool(x)


class GlobalAvgPool(mxnn.Module):
    def __call__(self, x: SparseTensor) -> mx.array:
        return global_avg_pool(x)


class GlobalMaxPool(mxnn.Module):
    def __call__(self, x: SparseTensor) -> mx.array:
        return global_max_pool(x)


# MARK: - helpers


def _init_kernel_params(
    module: mxnn.Module,
    input_dims: int,
    output_dims: int,
    kernel_volume: int,
    bias: bool,
) -> None:
    scale = math.sqrt(1.0 / (input_dims * kernel_volume))
    module.weight = mx.random.uniform(
        low=-scale,
        high=scale,
        shape=(kernel_volume, input_dims, output_dims),
    )
    if bias:
        module.bias = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims,),
        )


def _optional_bias(module: mxnn.Module) -> mx.array | None:
    return module.bias if 'bias' in module else None
