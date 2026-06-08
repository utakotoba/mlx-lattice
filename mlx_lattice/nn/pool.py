from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import KernelSpec, SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    pool3d,
    sum_pool3d,
)

PoolMode = Literal['sum', 'max', 'avg']

__all__ = [
    'AvgPool3d',
    'GlobalAvgPool',
    'GlobalMaxPool',
    'GlobalSumPool',
    'MaxPool3d',
    'Pool3d',
    'SumPool3d',
]


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
