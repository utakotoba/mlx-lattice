from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import KernelSpec, SparseTensor
from mlx_lattice.nn._export import (
    kernel_spec_attributes,
    lattice_module,
    path_attribute,
)
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


@lattice_module(
    'ops.pool3d',
    attributes=(
        path_attribute('mode', 'mode'),
        *kernel_spec_attributes(
            'kernel_size',
            'stride',
            'padding',
            'dilation',
        ),
    ),
)
class Pool3d(mxnn.Module):
    """Configurable local sparse 3D pooling module.

    ``mode`` selects ``sum``, ``max``, or ``avg`` reduction over a sparse kernel
    relation. The module returns a sparse tensor with output stride multiplied
    by the pooling stride.
    """

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


@lattice_module(
    'ops.sum_pool3d',
    attributes=kernel_spec_attributes(
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    ),
)
class SumPool3d(Pool3d):
    """Local sparse sum-pooling module."""

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


@lattice_module(
    'ops.max_pool3d',
    attributes=kernel_spec_attributes(
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    ),
)
class MaxPool3d(Pool3d):
    """Local sparse max-pooling module."""

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


@lattice_module(
    'ops.avg_pool3d',
    attributes=kernel_spec_attributes(
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    ),
)
class AvgPool3d(Pool3d):
    """Local sparse average-pooling module."""

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


@lattice_module('pool.global_sum')
class GlobalSumPool(mxnn.Module):
    """Batch-wise global sum-pooling module returning dense ``(B, C)`` rows."""

    def __call__(self, x: SparseTensor) -> mx.array:
        return global_sum_pool(x)


@lattice_module('pool.global_avg')
class GlobalAvgPool(mxnn.Module):
    """Batch-wise global average-pooling module returning dense ``(B, C)`` rows."""

    def __call__(self, x: SparseTensor) -> mx.array:
        return global_avg_pool(x)


@lattice_module('pool.global_max')
class GlobalMaxPool(mxnn.Module):
    """Batch-wise global max-pooling module returning dense ``(B, C)`` rows."""

    def __call__(self, x: SparseTensor) -> mx.array:
        return global_max_pool(x)
