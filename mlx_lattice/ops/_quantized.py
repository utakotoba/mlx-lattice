from __future__ import annotations

import mlx.core as mx

from mlx_lattice.core.quantized import QuantizedWeight


def quantized_matmul(
    feats: mx.array,
    weight: QuantizedWeight,
) -> mx.array:
    """Apply affine packed-weight matrix multiplication to feature rows."""
    if weight.kernel_size != (1, 1, 1):
        raise ValueError('quantized matmul requires a pointwise weight.')
    if feats.ndim != 2 or feats.shape[1] != weight.in_channels:
        raise ValueError(
            'features must have shape (N, quantized_weight.in_channels).'
        )
    if feats.dtype != weight.scales.dtype:
        raise ValueError('features must match quantized scale dtype.')
    if weight.storage_in_channels != weight.in_channels:
        feats = mx.pad(
            feats,
            [
                (0, 0),
                (0, weight.storage_in_channels - weight.in_channels),
            ],
        )
    return mx.quantized_matmul(
        feats,
        weight.weight[0],
        weight.scales[0],
        weight.biases[0],
        transpose=True,
        group_size=weight.group_size,
        bits=weight.bits,
        mode='affine',
    )
