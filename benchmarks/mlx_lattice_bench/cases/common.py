from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.core import SparseTensor


def param_grid(
    preset: str,
    *,
    small_rows: int = 256,
    standard_rows: int = 2048,
    full_rows: int = 8192,
    channels: tuple[int, ...] = (16, 64),
    batches: tuple[int, ...] = (1, 4),
) -> tuple[Mapping[str, Any], ...]:
    if preset == 'smoke':
        return (
            {'rows': small_rows, 'channels': channels[0], 'batches': 1},
        )
    if preset == 'standard':
        return tuple(
            {'rows': standard_rows, 'channels': channel, 'batches': batch}
            for channel in channels
            for batch in batches
        )
    if preset == 'full':
        return tuple(
            {'rows': rows, 'channels': channel, 'batches': batch}
            for rows in (standard_rows, full_rows)
            for channel in channels
            for batch in batches
        )
    raise ValueError("preset must be 'smoke', 'standard', or 'full'.")


@dataclass(frozen=True, slots=True)
class TensorInputs:
    x: SparseTensor


@dataclass(frozen=True, slots=True)
class WeightedInputs:
    x: SparseTensor
    weight: mx.array
    bias: mx.array | None = None
