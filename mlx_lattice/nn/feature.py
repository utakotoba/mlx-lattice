from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import feature as F
from mlx_lattice.ops.feature import GeluApprox

__all__ = [
    'GELU',
    'BatchNorm',
    'Dropout',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'RMSNorm',
    'ReLU',
    'SiLU',
    'Sigmoid',
    'Softplus',
    'Tanh',
]

if TYPE_CHECKING:
    from mlx_lattice.nn.quantized_feature import QuantizedLinear


class Linear(mxnn.Linear):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.linear(
            x,
            self.weight,
            self.bias if 'bias' in self else None,
        )

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = 'affine',
        quantize_input: bool = False,
    ) -> QuantizedLinear:
        from mlx_lattice.nn.quantized_feature import QuantizedLinear

        if quantize_input:
            raise ValueError(
                'affine sparse QuantizedLinear uses floating-point activations.'
            )
        return QuantizedLinear.from_linear(
            self,
            group_size=group_size,
            bits=4 if bits is None else bits,
            mode=mode,
        )


class ReLU(mxnn.ReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.relu(x)


class Sigmoid(mxnn.Sigmoid):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.sigmoid(x)


class GELU(mxnn.GELU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.gelu(x, approximate=cast('GeluApprox', self._approx))


class SiLU(mxnn.SiLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.silu(x)


class LeakyReLU(mxnn.LeakyReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.leaky_relu(x, negative_slope=self._negative_slope)


class Tanh(mxnn.Tanh):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.tanh(x)


class Softplus(mxnn.Softplus):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.softplus(x)


class Dropout(mxnn.Dropout):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.dropout(x, p=1 - self._p_1, training=self.training)


class BatchNorm(mxnn.BatchNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        mean = mx.mean(x.feats, axis=0)
        var = mx.var(x.feats, axis=0)
        if self.training and self.track_running_stats:
            mu = self.momentum
            self.running_mean = (1 - mu) * self.running_mean + mu * mean
            self.running_var = (1 - mu) * self.running_var + mu * var
        elif self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        return F.batch_norm(
            x,
            weight=self.weight if 'weight' in self else None,
            bias=self.bias if 'bias' in self else None,
            mean=mean,
            var=var,
            eps=self.eps,
        )


class LayerNorm(mxnn.LayerNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.layer_norm(
            x,
            weight=self.weight if 'weight' in self else None,
            bias=self.bias if 'bias' in self else None,
            eps=self.eps,
        )


class RMSNorm(mxnn.RMSNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.rms_norm(x, weight=self.weight, eps=self.eps)
