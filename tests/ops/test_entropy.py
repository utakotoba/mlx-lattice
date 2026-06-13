from __future__ import annotations

from itertools import pairwise

import mlx.core as mx

from mlx_lattice.ops import (
    normalized_cdf,
    range_decode,
    range_decode_from_prob,
    range_encode,
    range_encode_from_prob,
    rans_decode_from_prob,
    rans_encode_from_prob,
)


def test_normalized_cdf_is_strictly_monotonic() -> None:
    prob = mx.array(
        [
            [0.20, 0.30, 0.50],
            [0.10, 0.10, 0.80],
        ],
        dtype=mx.float32,
    )

    cdf = normalized_cdf(prob)
    mx.eval(cdf)
    rows = cdf.tolist()

    assert cdf.shape == (2, 4)
    for row in rows:
        unsigned = [value & 0xFFFF for value in row[:-1]] + [1 << 16]
        assert unsigned[0] == 0
        assert all(lhs < rhs for lhs, rhs in pairwise(unsigned))


def test_normalized_cdf_uses_probability_values() -> None:
    prob = mx.ones((1, 16), dtype=mx.float32) / 16

    cdf = normalized_cdf(prob)
    mx.eval(cdf)

    unsigned = [value & 0xFFFF for value in cdf.tolist()[0][:-1]]
    assert unsigned[:4] == [0, 4096, 8192, 12288]


def test_range_codec_roundtrip() -> None:
    prob = mx.array(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.70, 0.10],
            [0.10, 0.10, 0.10, 0.70],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=mx.float32,
    )
    symbols = mx.array([0, 1, 2, 3, 1], dtype=mx.int32)
    cdf = normalized_cdf(prob)

    stream = range_encode(cdf, symbols)
    decoded = range_decode(cdf, stream)
    mx.eval(decoded)

    assert stream.startswith(b'MLXGERNG01')
    assert decoded.tolist() == symbols.tolist()


def test_range_codec_prob_direct_roundtrip_matches_cdf_path() -> None:
    prob = mx.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.20, 0.30, 0.40],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=mx.float32,
    )
    symbols = mx.array([0, 3, 2], dtype=mx.int32)
    cdf = normalized_cdf(prob)

    cdf_stream = range_encode(cdf, symbols)
    prob_stream = range_encode_from_prob(prob, symbols)
    decoded = range_decode_from_prob(prob, prob_stream)
    mx.eval(decoded)

    assert prob_stream == cdf_stream
    assert decoded.tolist() == symbols.tolist()


def test_rans_codec_prob_direct_roundtrip() -> None:
    prob = mx.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.20, 0.30, 0.40],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=mx.float32,
    )
    symbols = mx.array([0, 3, 2], dtype=mx.int32)

    stream = rans_encode_from_prob(prob, symbols)
    decoded = rans_decode_from_prob(prob, stream)
    mx.eval(decoded)

    assert stream.startswith(b'MLXGERNS01')
    assert decoded.tolist() == symbols.tolist()


def test_range_codec_random_long_roundtrip() -> None:
    mx.random.seed(23)
    logits = mx.random.uniform(shape=(2048, 16), dtype=mx.float32)
    prob = logits / mx.sum(logits, axis=1, keepdims=True)
    symbols = mx.random.randint(0, 16, shape=(2048,), dtype=mx.int32)
    cdf = normalized_cdf(prob)

    stream = range_encode(cdf, symbols)
    decoded = range_decode(cdf, stream)
    mx.eval(decoded)

    assert decoded.tolist() == symbols.tolist()

    direct_stream = range_encode_from_prob(prob, symbols)
    direct_decoded = range_decode_from_prob(prob, direct_stream)
    mx.eval(direct_decoded)

    assert direct_decoded.tolist() == symbols.tolist()

    rans_stream = rans_encode_from_prob(prob, symbols)
    rans_decoded = rans_decode_from_prob(prob, rans_stream)
    mx.eval(rans_decoded)

    assert rans_decoded.tolist() == symbols.tolist()


def test_range_codec_empty_stream_roundtrip() -> None:
    prob = mx.zeros((0, 4), dtype=mx.float32)
    symbols = mx.array([], dtype=mx.int32)
    cdf = normalized_cdf(prob)

    stream = range_encode(cdf, symbols)
    decoded = range_decode(cdf, stream)
    mx.eval(decoded)

    assert decoded.shape == (0,)
