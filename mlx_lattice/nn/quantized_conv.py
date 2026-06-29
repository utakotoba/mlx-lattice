from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import (
    CoordinateMapKey,
    KernelSpec,
    QuantizedWeight,
    SparseTensor,
    quantize_weight,
)
from mlx_lattice.nn._export import kernel_spec_attributes, lattice_module
from mlx_lattice.nn.conv import (
    Conv3d,
    ConvTranspose3d,
    GenerativeConvTranspose3d,
    SubmConv3d,
)
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)

__all__ = [
    'QuantizedConv3d',
    'QuantizedConvTranspose3d',
    'QuantizedGenerativeConvTranspose3d',
    'QuantizedSubmConv3d',
]


class _QuantizedConvBase(mxnn.Module):
    spec: KernelSpec

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
        self.in_channels = quantized.in_channels
        self.out_channels = quantized.out_channels

    def _quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            self.weight,
            self.scales,
            self.biases,
            self.group_size,
            self.bits,
            self.in_channels,
            self.out_channels,
            self.spec.size,
            'dense_5d',
        )

    def _copy_from(
        self,
        source: mxnn.Module,
        group_size: int | None,
        bits: int,
    ) -> None:
        self.spec = source.spec
        self._assign_quantized(source.weight, group_size, bits)
        if 'bias' in source:
            self.bias = source.bias
        self.freeze()


@lattice_module(
    'sparse.quantized_conv3d',
    parameters=('weight', 'bias'),
    attributes=kernel_spec_attributes(
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    ),
)
class QuantizedConv3d(_QuantizedConvBase):
    """Affine weight-quantized sparse 3D convolution module.

    Weights are stored as packed int4/int8 affine ``QuantizedWeight`` metadata.
    Activations remain floating point. Coordinate semantics match
    :class:`mlx_lattice.nn.Conv3d`.
    """

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
        group_size: int | None = None,
        bits: int = 4,
    ) -> None:
        super().__init__()
        self._copy_from(
            Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            group_size,
            bits,
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
            self._quantized_weight(),
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
            coordinates=coordinates,
        )

    @classmethod
    def from_conv(
        cls,
        source: Conv3d,
        group_size: int | None = None,
        bits: int = 4,
    ) -> QuantizedConv3d:
        out = cls(
            source.weight.shape[4],
            source.weight.shape[0],
            kernel_size=source.spec.size,
            stride=source.spec.stride,
            padding=source.spec.padding,
            dilation=source.spec.dilation,
            bias='bias' in source,
            group_size=group_size,
            bits=bits,
        )
        out._copy_from(source, group_size, bits)
        return out


@lattice_module(
    'sparse.quantized_subm_conv3d',
    parameters=('weight', 'bias'),
    attributes=kernel_spec_attributes('kernel_size', 'dilation'),
)
class QuantizedSubmConv3d(_QuantizedConvBase):
    """Affine weight-quantized submanifold convolution module.

    Coordinate identity is preserved exactly as in ``SubmConv3d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
        group_size: int | None = None,
        bits: int = 4,
    ) -> None:
        super().__init__()
        self._copy_from(
            SubmConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
            ),
            group_size,
            bits,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return subm_conv3d(
            x,
            self._quantized_weight(),
            _optional_bias(self),
            kernel_size=self.spec.size,
            dilation=self.spec.dilation,
        )

    @classmethod
    def from_conv(
        cls,
        source: SubmConv3d,
        group_size: int | None = None,
        bits: int = 4,
    ) -> QuantizedSubmConv3d:
        out = cls(
            source.weight.shape[4],
            source.weight.shape[0],
            kernel_size=source.spec.size,
            dilation=source.spec.dilation,
            bias='bias' in source,
            group_size=group_size,
            bits=bits,
        )
        out._copy_from(source, group_size, bits)
        return out


@lattice_module(
    'sparse.quantized_conv_transpose3d',
    parameters=('weight', 'bias'),
    attributes=kernel_spec_attributes(
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    ),
)
class QuantizedConvTranspose3d(_QuantizedConvBase):
    """Affine weight-quantized sparse transpose-convolution module.

    Activations remain floating point and weight storage is packed affine
    int4/int8. Coordinate generation matches ``ConvTranspose3d``.
    """

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
        group_size: int | None = None,
        bits: int = 4,
    ) -> None:
        super().__init__()
        self._copy_from(
            ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            group_size,
            bits,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return conv_transpose3d(
            x,
            self._quantized_weight(),
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
            padding=self.spec.padding,
            dilation=self.spec.dilation,
        )

    @classmethod
    def from_conv(
        cls,
        source: ConvTranspose3d,
        group_size: int | None = None,
        bits: int = 4,
    ) -> QuantizedConvTranspose3d:
        out = cls(
            source.weight.shape[4],
            source.weight.shape[0],
            kernel_size=source.spec.size,
            stride=source.spec.stride,
            padding=source.spec.padding,
            dilation=source.spec.dilation,
            bias='bias' in source,
            group_size=group_size,
            bits=bits,
        )
        out._copy_from(source, group_size, bits)
        return out


@lattice_module(
    'sparse.quantized_generative_conv_transpose3d',
    parameters=('weight', 'bias'),
    attributes=kernel_spec_attributes('kernel_size', 'stride'),
)
class QuantizedGenerativeConvTranspose3d(_QuantizedConvBase):
    """Affine weight-quantized generative transpose-convolution module.

    The module stores packed affine weights and delegates coordinate generation
    to the generative transpose-convolution relation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        bias: bool = True,
        group_size: int | None = None,
        bits: int = 4,
    ) -> None:
        super().__init__()
        self._copy_from(
            GenerativeConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            ),
            group_size,
            bits,
        )

    def __call__(self, x: SparseTensor) -> SparseTensor:
        return generative_conv_transpose3d(
            x,
            self._quantized_weight(),
            _optional_bias(self),
            kernel_size=self.spec.size,
            stride=self.spec.stride,
        )

    @classmethod
    def from_conv(
        cls,
        source: GenerativeConvTranspose3d,
        group_size: int | None = None,
        bits: int = 4,
    ) -> QuantizedGenerativeConvTranspose3d:
        out = cls(
            source.weight.shape[4],
            source.weight.shape[0],
            kernel_size=source.spec.size,
            stride=source.spec.stride,
            bias='bias' in source,
            group_size=group_size,
            bits=bits,
        )
        out._copy_from(source, group_size, bits)
        return out


def _optional_bias(module: mxnn.Module) -> mx.array | None:
    return module.bias if 'bias' in module else None
