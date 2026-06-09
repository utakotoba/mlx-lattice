from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.core import SparseTensor


def param_grid(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
    small_n: int = 256,
    standard_n: int = 2048,
    full_n: int = 8192,
    channels: tuple[int, ...] = (16, 64),
    batches: tuple[int, ...] = (1, 4),
) -> tuple[Mapping[str, Any], ...]:
    if preset == 'smoke':
        sizes = n_values or (small_n,)
        return tuple(
            {'N': n, 'channels': channels[0], 'batches': 1} for n in sizes
        )
    if preset == 'standard':
        sizes = n_values or (standard_n,)
        return tuple(
            {'N': n, 'channels': channel, 'batches': batch}
            for n in sizes
            for channel in channels
            for batch in batches
        )
    if preset == 'full':
        sizes = n_values or (standard_n, full_n)
        return tuple(
            {'N': n, 'channels': channel, 'batches': batch}
            for n in sizes
            for channel in channels
            for batch in batches
        )
    raise ValueError("preset must be 'smoke', 'standard', or 'full'.")


def benchmark_n(params: Mapping[str, Any]) -> int:
    return int(params['N'])


@dataclass(frozen=True, slots=True)
class TensorInputs:
    x: SparseTensor


@dataclass(frozen=True, slots=True)
class WeightedInputs:
    x: SparseTensor
    weight: mx.array
    bias: mx.array | None = None
