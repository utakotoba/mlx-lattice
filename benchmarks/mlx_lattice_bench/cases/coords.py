from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.ops import (
    child_coords_from_indices,
    downsample_coords,
    intersection_coords,
    lookup_coords,
    morton_codes,
    morton_sort_coords,
    occupancy_downsample,
    occupancy_expand,
    union_coords,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class CoordInputs:
    lhs: mx.array
    rhs: mx.array


@dataclass(frozen=True, slots=True)
class OccupancyInputs:
    coords: mx.array
    downsampled_coords: mx.array
    occupancy: mx.array
    child_indices: mx.array


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(
            preset,
            n_values=n_values,
            channels=(8,),
            batches=(1,),
        )
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
        BenchmarkCase(
            name='morton_codes',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: morton_codes(inputs.lhs),
            units=('n_in',),
        ),
        BenchmarkCase(
            name='morton_sort_coords',
            group='coords',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: morton_sort_coords(inputs.lhs),
            units=('n_in',),
        ),
        BenchmarkCase(
            name='occupancy_downsample',
            group='coords',
            params=params,
            setup=_occupancy_setup,
            prepare=_occupancy_prepare,
            run=lambda inputs: occupancy_downsample(inputs.coords),
            units=('n_in',),
        ),
        BenchmarkCase(
            name='occupancy_expand',
            group='coords',
            params=params,
            setup=_occupancy_setup,
            prepare=_occupancy_prepare,
            run=lambda inputs: occupancy_expand(
                inputs.downsampled_coords, inputs.occupancy
            ),
            units=('n_in',),
        ),
        BenchmarkCase(
            name='child_coords_from_indices',
            group='coords',
            params=params,
            setup=_occupancy_setup,
            prepare=_occupancy_prepare,
            run=lambda inputs: child_coords_from_indices(
                inputs.coords, inputs.child_indices
            ),
            units=('n_in',),
        ),
    )


def _setup(params: Mapping[str, Any]) -> tuple[SparseArrays, SparseArrays]:
    rows = benchmark_n(params)
    lhs = sparse_arrays(rows=rows, channels=1)
    rhs = sparse_arrays(rows=rows, channels=1)
    return lhs, rhs


def _prepare(fixture: tuple[SparseArrays, SparseArrays]) -> CoordInputs:
    lhs, rhs = fixture
    return CoordInputs(lhs.coords, rhs.coords)


def _occupancy_setup(params: Mapping[str, Any]) -> OccupancyInputs:
    rows = benchmark_n(params)
    base = sparse_arrays(rows=rows, channels=1)
    downsampled = occupancy_downsample(base.coords)
    child_indices = mx.remainder(mx.arange(rows, dtype=mx.int32), 8)
    return OccupancyInputs(
        base.coords,
        downsampled.coords,
        downsampled.occupancy,
        child_indices,
    )


def _occupancy_prepare(fixture: OccupancyInputs) -> OccupancyInputs:
    return fixture
