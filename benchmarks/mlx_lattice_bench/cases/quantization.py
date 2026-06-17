from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lattice.core import SparseQuantization, SparseTensor
from mlx_lattice.core.coords.quantization import VoxelReduction
from mlx_lattice.ops import (
    sparse_quantize,
    voxelize,
    voxelize_with_quantization,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import PointArrays, point_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class PointInputs:
    points: mx.array
    feats: mx.array
    batch_indices: mx.array


@dataclass(frozen=True, slots=True)
class FixedVoxelFixture:
    points: PointArrays
    quantization: SparseQuantization
    template: SparseTensor


@dataclass(frozen=True, slots=True)
class FixedVoxelInputs:
    points: mx.array
    feats: mx.array
    quantization: SparseQuantization
    template: SparseTensor


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item)
        for item in param_grid(
            preset, n_values=n_values, channels=channels or (8, 32)
        )
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
        BenchmarkCase(
            name='voxelize_mean_fixed',
            group='quantization',
            params=params,
            setup=_setup_fixed,
            prepare=_prepare_fixed,
            run=lambda inputs: voxelize_with_quantization(
                inputs.quantization,
                inputs.feats,
                reduction='mean',
                template=inputs.template,
            ),
            compiled=_compiled_voxelize_fixed('mean'),
            backward=_backward_voxelize_fixed('mean'),
            units=('points', 'n_out', 'elements'),
        ),
        BenchmarkCase(
            name='voxelize_sum_fixed',
            group='quantization',
            params=params,
            setup=_setup_fixed,
            prepare=_prepare_fixed,
            run=lambda inputs: voxelize_with_quantization(
                inputs.quantization,
                inputs.feats,
                reduction='sum',
                template=inputs.template,
            ),
            compiled=_compiled_voxelize_fixed('sum'),
            backward=_backward_voxelize_fixed('sum'),
            units=('points', 'n_out', 'elements'),
        ),
    )


def _setup(params: Mapping[str, Any]) -> PointArrays:
    return point_arrays(
        rows=benchmark_n(params),
        channels=int(params['channels']),
        batches=int(params['batches']),
    )


def _prepare(fixture: PointArrays) -> PointInputs:
    return PointInputs(fixture.points, fixture.feats, fixture.batch_indices)


def _setup_fixed(params: Mapping[str, Any]) -> FixedVoxelFixture:
    points = _setup(params)
    quantization = sparse_quantize(
        points.points,
        voxel_size=1.0,
        batch_indices=points.batch_indices,
    )
    template = SparseTensor(
        quantization.coords,
        mx.zeros_like(points.feats),
        active_rows=quantization.active_rows,
    )
    return FixedVoxelFixture(points, quantization, template)


def _prepare_fixed(fixture: FixedVoxelFixture) -> FixedVoxelInputs:
    return FixedVoxelInputs(
        fixture.points.points,
        fixture.points.feats,
        fixture.quantization,
        fixture.template,
    )


def _compiled_voxelize(reduction: VoxelReduction) -> Any:
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


def _compiled_voxelize_fixed(reduction: VoxelReduction) -> Any:
    def factory(fixture: FixedVoxelFixture) -> tuple[Any, tuple[Any, ...]]:
        def fn(feats: mx.array) -> mx.array:
            return voxelize_with_quantization(
                fixture.quantization,
                feats,
                reduction=reduction,
                template=fixture.template,
            ).feats

        return fn, (fixture.points.feats,)

    return factory


def _backward_voxelize(reduction: VoxelReduction) -> Any:
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


def _backward_voxelize_fixed(reduction: VoxelReduction) -> Any:
    def factory(fixture: FixedVoxelFixture) -> tuple[Any, tuple[Any, ...]]:
        def loss(feats: mx.array) -> mx.array:
            out = voxelize_with_quantization(
                fixture.quantization,
                feats,
                reduction=reduction,
                template=fixture.template,
            )
            return mx.sum(out.feats)

        return mx.grad(loss), (fixture.points.feats,)

    return factory
