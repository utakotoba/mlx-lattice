from __future__ import annotations

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import QuantizedWeight, SparseTensor, quantize_weight
from mlx_lattice.ops import feature as F

__all__ = ['QuantizedLinear']


class QuantizedLinear(mxnn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int | None = None,
        bits: int = 4,
        mode: str = 'affine',
    ) -> None:
        super().__init__()
        _require_affine(mode)
        from mlx_lattice.nn.feature import Linear

        source = Linear(input_dims, output_dims, bias=bias)
        self._assign_quantized(source.weight, group_size, bits)
        if bias:
            self.bias = source.bias
        self.freeze()

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return F.linear(
            x,
            self._quantized_weight(),
            self.bias if 'bias' in self else None,
        )

    @classmethod
    def from_linear(
        cls,
        linear: mxnn.Module,
        group_size: int | None = None,
        bits: int = 4,
        mode: str = 'affine',
    ) -> QuantizedLinear:
        _require_affine(mode)
        output_dims, input_dims = map(int, linear.weight.shape)
        out = cls(
            input_dims,
            output_dims,
            bias='bias' in linear,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        out._assign_quantized(linear.weight, group_size, bits)
        if 'bias' in linear:
            out.bias = linear.bias
        out.freeze()
        return out

    def _assign_quantized(
        self,
        weight: mx.array,
        group_size: int | None,
        bits: int,
    ) -> None:
        quantized = quantize_weight(
            weight, group_size=group_size, bits=bits
        )
        self.weight = quantized.weight
        self.scales = quantized.scales
        self.biases = quantized.biases
        self.group_size = quantized.group_size
        self.bits = quantized.bits
        self.input_dims = quantized.in_channels
        self.output_dims = quantized.out_channels

    def _quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            self.weight,
            self.scales,
            self.biases,
            self.group_size,
            self.bits,
            self.input_dims,
            self.output_dims,
            (1, 1, 1),
            'linear',
        )


def _require_affine(mode: str) -> None:
    if mode != 'affine':
        raise ValueError('QuantizedLinear currently supports affine mode.')
