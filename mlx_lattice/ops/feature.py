from __future__ import annotations

from typing import Literal

import mlx.core as mx

from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.ops._quantized import quantized_matmul

GeluApprox = Literal['none', 'precise', 'tanh', 'fast']

__all__ = [
    'batch_norm',
    'dropout',
    'gelu',
    'layer_norm',
    'leaky_relu',
    'linear',
    'relu',
    'rms_norm',
    'sigmoid',
    'silu',
    'softplus',
    'tanh',
]


def linear(
    x: SparseTensor,
    weight: mx.array | QuantizedWeight,
    bias: mx.array | None = None,
) -> SparseTensor:
    if isinstance(weight, QuantizedWeight):
        feats = quantized_matmul(x.feats, weight)
        return x.replace(feats=_with_bias(feats, bias))
    _require_2d_weight(weight)
    if weight.shape[1] != x.channels:
        raise ValueError('weight input channels must match x.channels.')
    feats = x.feats @ weight.T
    return x.replace(feats=_with_bias(feats, bias))


def relu(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.maximum(x.feats, 0))


def sigmoid(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.sigmoid(x.feats))


def gelu(
    x: SparseTensor,
    *,
    approximate: GeluApprox = 'none',
) -> SparseTensor:
    if approximate in ('none', 'precise'):
        scale = mx.array(0.5, dtype=x.feats.dtype)
        root_half = mx.array(0.7071067811865476, dtype=x.feats.dtype)
        return x.replace(
            feats=scale * x.feats * (1 + mx.erf(x.feats * root_half))
        )
    if approximate == 'tanh':
        coeff = mx.array(0.044715, dtype=x.feats.dtype)
        scale = mx.array(0.7978845608028654, dtype=x.feats.dtype)
        return x.replace(
            feats=0.5
            * x.feats
            * (1 + mx.tanh(scale * (x.feats + coeff * x.feats**3)))
        )
    if approximate == 'fast':
        return x.replace(feats=x.feats * mx.sigmoid(1.702 * x.feats))
    raise ValueError(
        "approximate must be 'none', 'precise', 'tanh', or 'fast'."
    )


def silu(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=x.feats * mx.sigmoid(x.feats))


def leaky_relu(
    x: SparseTensor,
    *,
    negative_slope: float = 0.01,
) -> SparseTensor:
    slope = mx.array(float(negative_slope), dtype=x.feats.dtype)
    return x.replace(feats=mx.where(x.feats >= 0, x.feats, x.feats * slope))


def tanh(x: SparseTensor) -> SparseTensor:
    return x.replace(feats=mx.tanh(x.feats))


def softplus(
    x: SparseTensor,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> SparseTensor:
    if beta <= 0:
        raise ValueError('beta must be positive.')
    scaled = x.feats * beta
    feats = mx.where(
        scaled > threshold,
        x.feats,
        mx.log(1 + mx.exp(scaled)) / beta,
    )
    return x.replace(feats=feats)


def dropout(
    x: SparseTensor,
    *,
    p: float = 0.5,
    training: bool = True,
) -> SparseTensor:
    if p < 0 or p >= 1:
        raise ValueError('p must satisfy 0 <= p < 1.')
    if not training or p == 0:
        return x.replace(feats=x.feats)
    keep = 1.0 - p
    mask = mx.random.bernoulli(p=keep, shape=x.feats.shape)
    return x.replace(feats=x.feats * mask.astype(x.feats.dtype) / keep)


def batch_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    mean: mx.array | None = None,
    var: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    if eps <= 0:
        raise ValueError('eps must be positive.')
    mean = mx.mean(x.feats, axis=0) if mean is None else mean
    var = mx.var(x.feats, axis=0) if var is None else var
    _require_channel_vector(mean, x.channels, 'mean')
    _require_channel_vector(var, x.channels, 'var')
    feats = (x.feats - mean) * mx.rsqrt(var + eps)
    return x.replace(feats=_affine(feats, weight=weight, bias=bias))


def layer_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    if eps <= 0:
        raise ValueError('eps must be positive.')
    if weight is not None:
        _require_channel_vector(weight, x.channels, 'weight')
    if bias is not None:
        _require_channel_vector(bias, x.channels, 'bias')
    return x.replace(feats=mx.fast.layer_norm(x.feats, weight, bias, eps))


def rms_norm(
    x: SparseTensor,
    *,
    weight: mx.array | None = None,
    eps: float = 1e-5,
) -> SparseTensor:
    if eps <= 0:
        raise ValueError('eps must be positive.')
    if weight is not None:
        _require_channel_vector(weight, x.channels, 'weight')
    return x.replace(feats=mx.fast.rms_norm(x.feats, weight, eps))


# MARK: - helpers


def _affine(
    feats: mx.array,
    *,
    weight: mx.array | None,
    bias: mx.array | None,
) -> mx.array:
    if weight is not None:
        _require_channel_vector(weight, int(feats.shape[1]), 'weight')
        feats = feats * weight
    return _with_bias(feats, bias)


def _with_bias(feats: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return feats
    _require_channel_vector(bias, int(feats.shape[1]), 'bias')
    return feats + bias


def _require_2d_weight(weight: mx.array) -> None:
    if weight.ndim != 2:
        raise ValueError('weight must have shape (C_out, C_in).')


def _require_channel_vector(
    value: mx.array,
    channels: int,
    name: str,
) -> None:
    if value.ndim != 1 or value.shape[0] != channels:
        raise ValueError(f'{name} must have shape ({channels},).')
