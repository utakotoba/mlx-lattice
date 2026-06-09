from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.ops import sparse_quantize, voxelize

from mlx_lattice_bench.cases.common import param_grid
from mlx_lattice_bench.datasets import PointArrays, point_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class PointInputs:
    points: mx.array
    feats: mx.array
    batch_indices: mx.array


def cases(preset: str) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        {**dict(item), 'points': item['rows']}
        for item in param_grid(preset, channels=(8, 32))
    )
    return (
        BenchmarkCase(
            name='sparse_quantize',
            group='quantization',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: sparse_quantize(
                inputs.points,
                voxel_size=1.0,
                batch_indices=inputs.batch_indices,
            ),
            units=('points', 'n_out'),
        ),
        BenchmarkCase(
            name='voxelize_mean',
            group='quantization',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: voxelize(
                inputs.points,
                inputs.feats,
                voxel_size=1.0,
                batch_indices=inputs.batch_indices,
                reduction='mean',
            ),
            compiled=_compiled_voxelize('mean'),
            backward=_backward_voxelize('mean'),
            units=('points', 'n_out', 'elements'),
        ),
        BenchmarkCase(
            name='voxelize_sum',
            group='quantization',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: voxelize(
                inputs.points,
                inputs.feats,
                voxel_size=1.0,
                batch_indices=inputs.batch_indices,
                reduction='sum',
            ),
            compiled=_compiled_voxelize('sum'),
            backward=_backward_voxelize('sum'),
            units=('points', 'n_out', 'elements'),
        ),
    )


def _setup(params: Mapping[str, Any]) -> PointArrays:
    return point_arrays(
        rows=int(params['rows']),
        channels=int(params['channels']),
        batches=int(params['batches']),
    )


def _prepare(fixture: PointArrays) -> PointInputs:
    return PointInputs(fixture.points, fixture.feats, fixture.batch_indices)


def _compiled_voxelize(
    reduction: str,
) -> Any:
    def factory(fixture: PointArrays) -> tuple[Any, tuple[Any, ...]]:
        def fn(feats: mx.array) -> mx.array:
            return voxelize(
                fixture.points,
                feats,
                voxel_size=1.0,
                batch_indices=fixture.batch_indices,
                reduction=reduction,
            ).feats

        return fn, (fixture.feats,)

    return factory


def _backward_voxelize(
    reduction: str,
) -> Any:
    def factory(fixture: PointArrays) -> tuple[Any, tuple[Any, ...]]:
        def loss(feats: mx.array) -> mx.array:
            out = voxelize(
                fixture.points,
                feats,
                voxel_size=1.0,
                batch_indices=fixture.batch_indices,
                reduction=reduction,
            )
            return mx.sum(out.feats)

        return mx.grad(loss), (fixture.feats,)

    return factory
