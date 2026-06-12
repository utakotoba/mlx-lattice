from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    sum_pool3d,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase

type PoolKind = Literal[
    'sum',
    'max',
    'avg',
    'global_sum',
    'global_max',
    'global_avg',
]


@dataclass(frozen=True, slots=True)
class PoolInputs:
    x: SparseTensor


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
    return tuple(
        _case(name, kind, params)
        for name, kind in (
            ('sum_pool3d', 'sum'),
            ('max_pool3d', 'max'),
            ('avg_pool3d', 'avg'),
            ('global_sum_pool', 'global_sum'),
            ('global_max_pool', 'global_max'),
            ('global_avg_pool', 'global_avg'),
        )
    )


def _case(
    name: str,
    kind: PoolKind,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='pool',
        params=params,
        setup=_setup,
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs.x),
        compiled=_compiled(kind),
        backward=_backward(kind),
        units=('elements', 'n_in', 'n_out'),
    )


def _setup(params: Mapping[str, Any]) -> SparseArrays:
    return sparse_arrays(
        rows=benchmark_n(params),
        channels=int(params['channels']),
        batches=int(params['batches']),
    )


def _prepare(fixture: SparseArrays) -> PoolInputs:
    return PoolInputs(fixture.tensor())


def _run(kind: PoolKind, x: SparseTensor) -> Any:
    if kind == 'sum':
        return sum_pool3d(x, kernel_size=2, stride=2)
    if kind == 'max':
        return max_pool3d(x, kernel_size=2, stride=2)
    if kind == 'avg':
        return avg_pool3d(x, kernel_size=2, stride=2)
    if kind == 'global_sum':
        return global_sum_pool(x)
    if kind == 'global_max':
        return global_max_pool(x)
    return global_avg_pool(x)


def _compiled(
    kind: PoolKind,
) -> Callable[[SparseArrays], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: SparseArrays) -> tuple[Any, tuple[Any, ...]]:
        base = fixture.tensor()

        def fn(feats: mx.array) -> Any:
            x = base.replace(feats=feats)
            return (
                _run(kind, x).feats
                if kind in ('sum', 'max', 'avg')
                else _run(kind, x)
            )

        return fn, (fixture.feats,)

    return factory


def _backward(
    kind: PoolKind,
) -> Callable[[SparseArrays], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: SparseArrays) -> tuple[Any, tuple[Any, ...]]:
        base = fixture.tensor()

        def loss(feats: mx.array) -> mx.array:
            x = base.replace(feats=feats)
            out = _run(kind, x)
            values = out.feats if kind in ('sum', 'max', 'avg') else out
            return mx.sum(values)

        return mx.grad(loss), (fixture.feats,)

    return factory
