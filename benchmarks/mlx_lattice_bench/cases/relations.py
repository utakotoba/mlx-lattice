from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    generative_kernel_relation,
    kernel_relation,
    knn_relation,
    radius_relation,
    transposed_kernel_relation,
)

from mlx_lattice_bench.cases.common import param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class RelationInputs:
    x: SparseTensor
    transposed: SparseTensor


def cases(preset: str) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(preset, channels=(8,), batches=(1,))
    )
    return (
        BenchmarkCase(
            name='kernel_relation',
            group='relations',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: kernel_relation(
                inputs.x,
                kernel_size=(3, 3, 3),
            ),
            units=('edges', 'n_in'),
        ),
        BenchmarkCase(
            name='generative_kernel_relation',
            group='relations',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: generative_kernel_relation(
                inputs.x,
                kernel_size=(2, 2, 2),
                stride=2,
            ),
            units=('edges', 'n_in'),
        ),
        BenchmarkCase(
            name='transposed_kernel_relation',
            group='relations',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: transposed_kernel_relation(
                inputs.transposed,
                kernel_size=(2, 2, 2),
                stride=2,
            ),
            units=('edges', 'n_in'),
        ),
        BenchmarkCase(
            name='knn_relation',
            group='relations',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: knn_relation(inputs.x, k=8),
            units=('edges', 'n_in'),
        ),
        BenchmarkCase(
            name='radius_relation',
            group='relations',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: radius_relation(inputs.x, radius=2.0),
            units=('edges', 'n_in'),
        ),
    )


def _setup(params: Mapping[str, Any]) -> SparseArrays:
    return sparse_arrays(rows=int(params['rows']), channels=1)


def _prepare(fixture: SparseArrays) -> RelationInputs:
    return RelationInputs(fixture.tensor(), fixture.tensor(stride=2))
