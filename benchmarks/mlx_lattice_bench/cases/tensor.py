from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import prune_mask

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class TensorInputs:
    x: SparseTensor
    mask: mx.array


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(
            preset, n_values=n_values, channels=channels or (16, 64)
        )
    )
    return (
        BenchmarkCase(
            name='prune_mask',
            group='tensor',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: prune_mask(inputs.x, inputs.mask),
            units=('elements', 'n_in'),
        ),
    )


def _setup(params: Mapping[str, Any]) -> tuple[SparseArrays, mx.array]:
    rows = benchmark_n(params)
    arrays = sparse_arrays(rows=rows, channels=int(params['channels']))
    mask = cast(
        mx.array,
        mx.remainder(mx.arange(rows, dtype=mx.int32), 2) == 0,
    )
    return arrays, mask


def _prepare(fixture: tuple[SparseArrays, mx.array]) -> TensorInputs:
    arrays, mask = fixture
    return TensorInputs(arrays.tensor(), mask)
