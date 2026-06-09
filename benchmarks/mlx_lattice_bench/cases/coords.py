from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.ops import (
    downsample_coords,
    intersection_coords,
    lookup_coords,
    union_coords,
)

from mlx_lattice_bench.cases.common import param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class CoordInputs:
    lhs: mx.array
    rhs: mx.array


def cases(preset: str) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(preset, channels=(8,), batches=(1,))
    )
    return (
        BenchmarkCase(
            name='downsample_coords',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: downsample_coords(inputs.lhs, stride=2),
            units=('n_in', 'n_out'),
        ),
        BenchmarkCase(
            name='union_coords',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: union_coords(inputs.lhs, inputs.rhs),
            units=('n_in', 'n_out'),
        ),
        BenchmarkCase(
            name='intersection_coords',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: intersection_coords(inputs.lhs, inputs.rhs),
            units=('n_in', 'n_out'),
        ),
        BenchmarkCase(
            name='lookup_coords',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: lookup_coords(inputs.lhs, inputs.rhs),
            units=('n_in', 'n_out'),
        ),
    )


def _setup(params: Mapping[str, Any]) -> tuple[SparseArrays, SparseArrays]:
    rows = int(params['rows'])
    lhs = sparse_arrays(rows=rows, channels=1)
    rhs = sparse_arrays(rows=rows, channels=1)
    return lhs, rhs


def _prepare(fixture: tuple[SparseArrays, SparseArrays]) -> CoordInputs:
    lhs, rhs = fixture
    return CoordInputs(lhs.coords, rhs.coords)
