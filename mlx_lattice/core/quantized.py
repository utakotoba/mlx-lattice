from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_lattice.core.types import Triple

QuantizedWeightLayout = Literal['linear', 'kernel_major', 'dense_5d']

__all__ = [
    'QuantizedWeight',
    'QuantizedWeightLayout',
    'dequantize_weight',
    'quantize_weight',
]


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """Packed affine INT4/INT8 inference weight.

    The object stores packed ``uint32`` integer codes plus per-group affine
    ``scales`` and ``biases``. Logical values are reconstructed as
    ``scale * code + bias`` by quantized linear and convolution paths.

    ``layout`` records the logical source shape:
    ``linear`` for ``(C_out, C_in)``, ``kernel_major`` for
    ``(K, C_in, C_out)``, and ``dense_5d`` for
    ``(C_out, Kx, Ky, Kz, C_in)``.
    """

    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int
    in_channels: int
    out_channels: int
    kernel_size: Triple
    layout: QuantizedWeightLayout

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError('quantized weight bits must be 4 or 8.')
        if self.group_size not in (32, 64, 128):
            raise ValueError(
                'quantized weight group_size must be 32, 64, or 128.'
            )
        if self.weight.dtype != mx.uint32 or self.weight.ndim != 3:
            raise ValueError(
                'packed weight must be a three-dimensional uint32 array.'
            )
        if self.scales.dtype not in (mx.float16, mx.float32):
            raise ValueError('quantized scales must be float16 or float32.')
        if self.biases.dtype != self.scales.dtype:
            raise ValueError('quantized biases must match scales dtype.')
        if self.scales.ndim != 3 or self.biases.shape != self.scales.shape:
            raise ValueError(
                'scales and biases must have shape (K, C_out, G).'
            )
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError('quantized weight channels must be positive.')
        if any(size <= 0 for size in self.kernel_size):
            raise ValueError(
                'quantized weight kernel dimensions must be positive.'
            )
        kernel_rows = _volume(self.kernel_size)
        if self.weight.shape[0] != kernel_rows:
            raise ValueError(
                'packed weight kernel rows do not match kernel_size.'
            )
        if self.is_pointwise:
            if self.weight.shape[1] != self.out_channels:
                raise ValueError(
                    'pointwise packed weight output channels do not match.'
                )
            if self.scales.shape[:2] != self.weight.shape[:2]:
                raise ValueError(
                    'pointwise weight and quantization rows must match.'
                )
        else:
            if self.weight.shape[2] != self.out_channels:
                raise ValueError(
                    'spatial packed weight output channels do not match.'
                )
            if self.scales.shape[0] != self.weight.shape[0] or (
                self.scales.shape[2] != self.out_channels
            ):
                raise ValueError(
                    'spatial weight and quantization rows must match.'
                )
        if self.storage_in_channels % self.group_size != 0:
            raise ValueError(
                'packed storage channels must be divisible by group_size.'
            )
        group_dim = 2 if self.is_pointwise else 1
        if (
            self.scales.shape[group_dim]
            != self.storage_in_channels // self.group_size
        ):
            raise ValueError(
                'quantization group count does not match storage.'
            )
        if self.storage_in_channels < self.in_channels:
            raise ValueError(
                'packed storage channels cannot be smaller than logical channels.'
            )

    @property
    def storage_in_channels(self) -> int:
        packed_dim = 2 if self.is_pointwise else 1
        return int(self.weight.shape[packed_dim]) * 32 // self.bits

    @property
    def is_pointwise(self) -> bool:
        return _volume(self.kernel_size) == 1

    @property
    def nbytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes + self.biases.nbytes


def quantize_weight(
    weight: mx.array,
    *,
    group_size: int | None = None,
    bits: int = 4,
) -> QuantizedWeight:
    """Pack a linear or sparse-convolution weight for inference.

    Args:
        weight: Floating ``float16`` or ``float32`` weight. Accepted shapes are
            ``(C_out, C_in)``, ``(K, C_in, C_out)``, or
            ``(C_out, Kx, Ky, Kz, C_in)``.
        group_size: Quantization group size. ``None`` chooses ``64`` for
            ``C_in >= 64`` and ``32`` otherwise.
        bits: Packed integer width, either ``4`` or ``8``.

    Returns:
        A ``QuantizedWeight`` containing packed storage and affine metadata.
        Input channels are padded in storage to the selected group size when
        needed; logical ``in_channels`` remains the original channel count.
    """

    if weight.dtype not in (mx.float16, mx.float32):
        raise ValueError(
            'weight quantization supports float16 and float32.'
        )
    if bits not in (4, 8):
        raise ValueError('weight quantization supports only 4 or 8 bits.')

    rows, layout, kernel_size, in_channels, out_channels = _weight_rows(
        weight
    )
    group_size = _resolve_group_size(in_channels, group_size)
    storage_in_channels = _round_up(in_channels, group_size)
    if storage_in_channels != in_channels:
        rows = mx.pad(
            rows, [(0, 0), (0, storage_in_channels - in_channels)]
        )

    packed, scales, biases = mx.quantize(
        rows,
        group_size=group_size,
        bits=bits,
        mode='affine',
    )
    kernel_rows = _volume(kernel_size)
    packed = packed.reshape((kernel_rows, out_channels, -1))
    scales = scales.reshape((kernel_rows, out_channels, -1))
    biases = biases.reshape((kernel_rows, out_channels, -1))
    if kernel_rows > 1:
        packed = mx.contiguous(packed.transpose(0, 2, 1))
        scales = mx.contiguous(scales.transpose(0, 2, 1))
        biases = mx.contiguous(biases.transpose(0, 2, 1))
    return QuantizedWeight(
        weight=packed,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        layout=layout,
    )


def dequantize_weight(weight: QuantizedWeight) -> mx.array:
    """Restore the logical floating-point weight represented by ``weight``.

    The returned array uses the original logical layout recorded by
    ``weight.layout`` and slices away any padded storage channels.
    """

    kernel_rows = _volume(weight.kernel_size)
    packed = weight.weight
    scales = weight.scales
    biases = weight.biases
    if not weight.is_pointwise:
        packed = packed.transpose(0, 2, 1)
        scales = scales.transpose(0, 2, 1)
        biases = biases.transpose(0, 2, 1)
    rows = mx.dequantize(
        packed.reshape((kernel_rows * weight.out_channels, -1)),
        scales.reshape((kernel_rows * weight.out_channels, -1)),
        biases.reshape((kernel_rows * weight.out_channels, -1)),
        group_size=weight.group_size,
        bits=weight.bits,
        mode='affine',
    )[:, : weight.in_channels]
    if weight.layout == 'linear':
        return rows.reshape((weight.out_channels, weight.in_channels))
    mapped = rows.reshape(
        (kernel_rows, weight.out_channels, weight.in_channels)
    )
    if weight.layout == 'kernel_major':
        return mapped.transpose(0, 2, 1)
    return mapped.reshape(
        (*weight.kernel_size, weight.out_channels, weight.in_channels)
    ).transpose(3, 0, 1, 2, 4)


def _weight_rows(
    weight: mx.array,
) -> tuple[mx.array, QuantizedWeightLayout, Triple, int, int]:
    if weight.ndim == 2:
        out_channels, in_channels = map(int, weight.shape)
        return (
            weight,
            'linear',
            (1, 1, 1),
            in_channels,
            out_channels,
        )
    if weight.ndim == 3:
        kernel_rows, in_channels, out_channels = map(int, weight.shape)
        return (
            weight.transpose(0, 2, 1).reshape(
                (kernel_rows * out_channels, in_channels)
            ),
            'kernel_major',
            (kernel_rows, 1, 1),
            in_channels,
            out_channels,
        )
    if weight.ndim == 5:
        out_channels, kx, ky, kz, in_channels = map(int, weight.shape)
        return (
            weight.transpose(1, 2, 3, 0, 4).reshape(
                (kx * ky * kz * out_channels, in_channels)
            ),
            'dense_5d',
            (kx, ky, kz),
            in_channels,
            out_channels,
        )
    raise ValueError(
        'weight must have shape (C_out, C_in), (K, C_in, C_out), '
        'or (C_out, Kx, Ky, Kz, C_in).'
    )


def _resolve_group_size(in_channels: int, group_size: int | None) -> int:
    if group_size is None:
        return 64 if in_channels >= 64 else 32
    if group_size not in (32, 64, 128):
        raise ValueError('group_size must be 32, 64, or 128.')
    return group_size


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _volume(size: Triple) -> int:
    return size[0] * size[1] * size[2]
