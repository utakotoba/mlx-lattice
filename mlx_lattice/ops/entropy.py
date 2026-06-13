from __future__ import annotations

import mlx.core as mx

from mlx_lattice import _ext


def normalized_cdf(prob: mx.array) -> mx.array:
    """Convert probability rows to int16 normalized CDF rows."""
    if prob.ndim != 2:
        raise ValueError('prob must be a two-dimensional array.')
    return _ext.normalized_cdf(prob)


def range_encode(cdf: mx.array, symbols: mx.array) -> bytes:
    """Encode int32 symbols using int16 normalized CDF rows."""
    if cdf.ndim != 2:
        raise ValueError('cdf must be a two-dimensional array.')
    if symbols.ndim != 1:
        raise ValueError('symbols must be a one-dimensional array.')
    return _ext.range_encode(cdf, symbols.astype(mx.int32))


def range_decode(cdf: mx.array, stream: bytes) -> mx.array:
    """Decode int32 symbols from a range-coded stream."""
    if cdf.ndim != 2:
        raise ValueError('cdf must be a two-dimensional array.')
    return _ext.range_decode(cdf, stream)


def range_encode_from_prob(prob: mx.array, symbols: mx.array) -> bytes:
    """Encode int32 symbols directly from probability rows."""
    if prob.ndim != 2:
        raise ValueError('prob must be a two-dimensional array.')
    if symbols.ndim != 1:
        raise ValueError('symbols must be a one-dimensional array.')
    return _ext.range_encode_from_prob(prob, symbols.astype(mx.int32))


def range_decode_from_prob(prob: mx.array, stream: bytes) -> mx.array:
    """Decode int32 symbols directly from probability rows."""
    if prob.ndim != 2:
        raise ValueError('prob must be a two-dimensional array.')
    return _ext.range_decode_from_prob(prob, stream)


def rans_encode_from_prob(prob: mx.array, symbols: mx.array) -> bytes:
    """Encode int32 symbols with byte-oriented rANS from probability rows."""
    if prob.ndim != 2:
        raise ValueError('prob must be a two-dimensional array.')
    if symbols.ndim != 1:
        raise ValueError('symbols must be a one-dimensional array.')
    return _ext.rans_encode_from_prob(prob, symbols.astype(mx.int32))


def rans_decode_from_prob(prob: mx.array, stream: bytes) -> mx.array:
    """Decode int32 symbols with byte-oriented rANS from probability rows."""
    if prob.ndim != 2:
        raise ValueError('prob must be a two-dimensional array.')
    return _ext.rans_decode_from_prob(prob, stream)
