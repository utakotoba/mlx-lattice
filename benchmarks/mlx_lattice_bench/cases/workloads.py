from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    conv3d,
    knn_relation,
    layer_norm,
    linear,
    relu,
    sparse_quantize,
    subm_conv3d,
    sum_pool3d,
    voxelize,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import (
    PointArrays,
    SparseArrays,
    dense_bias,
    dense_weight,
    point_arrays,
    sparse_arrays,
)
from mlx_lattice_bench.harness import BenchmarkCase

type WorkloadKind = Literal[
    'voxel_stem',
    'subm_block',
    'downsample_block',
    'mini_encoder',
    'neighbor_pipeline',
]


@dataclass(frozen=True, slots=True)
class WorkloadFixture:
    sparse: SparseArrays
    points: PointArrays | None
    linear_weight: mx.array
    bias: mx.array
    kernel3_weight: mx.array


@dataclass(frozen=True, slots=True)
class WorkloadInputs:
    x: SparseTensor
    points: mx.array | None
    point_feats: mx.array | None
    batch_indices: mx.array | None
    linear_weight: mx.array
    bias: mx.array
    kernel3_weight: mx.array


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        dict(item) for item in param_grid(preset, n_values=n_values)
    )
    return tuple(
        _case(name, kind, params)
        for name, kind in (
            ('workload_voxel_stem', 'voxel_stem'),
            ('workload_subm_block', 'subm_block'),
            ('workload_downsample_block', 'downsample_block'),
            ('workload_mini_encoder', 'mini_encoder'),
            ('workload_neighbor_pipeline', 'neighbor_pipeline'),
        )
    )


def _case(
    name: str,
    kind: WorkloadKind,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='workloads',
        params=params,
        setup=lambda item: _setup(kind, item),
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs),
        compiled=_compiled(kind) if kind != 'neighbor_pipeline' else None,
        backward=_backward(kind) if kind != 'neighbor_pipeline' else None,
        units=('edges', 'elements', 'points', 'n_in', 'n_out'),
    )


def _setup(
    kind: WorkloadKind, params: Mapping[str, Any]
) -> WorkloadFixture:
    channels = int(params['channels'])
    rows = benchmark_n(params)
    batches = int(params['batches'])
    point_workload = kind in _POINT_WORKLOADS
    return WorkloadFixture(
        sparse=sparse_arrays(rows=rows, channels=channels, batches=batches),
        points=point_arrays(
            rows=rows,
            channels=channels,
            batches=batches,
        )
        if point_workload
        else None,
        linear_weight=dense_weight((channels, channels)),
        bias=dense_bias(channels),
        kernel3_weight=dense_weight((channels, 3, 3, 3, channels)),
    )


def _prepare(fixture: WorkloadFixture) -> WorkloadInputs:
    return WorkloadInputs(
        x=fixture.sparse.tensor(),
        points=None if fixture.points is None else fixture.points.points,
        point_feats=None
        if fixture.points is None
        else fixture.points.feats,
        batch_indices=None
        if fixture.points is None
        else fixture.points.batch_indices,
        linear_weight=fixture.linear_weight,
        bias=fixture.bias,
        kernel3_weight=fixture.kernel3_weight,
    )


def _run(kind: WorkloadKind, inputs: WorkloadInputs) -> Any:
    if kind == 'voxel_stem':
        points, point_feats, batch_indices = _point_inputs(inputs)
        x = voxelize(
            points,
            point_feats,
            batch_indices=batch_indices,
        )
        return relu(linear(x, inputs.linear_weight, inputs.bias))
    if kind == 'subm_block':
        y = subm_conv3d(inputs.x, inputs.kernel3_weight, kernel_size=3)
        y = relu(layer_norm(y))
        return subm_conv3d(y, inputs.kernel3_weight, kernel_size=3)
    if kind == 'downsample_block':
        y = conv3d(
            inputs.x,
            inputs.kernel3_weight,
            kernel_size=3,
            stride=2,
        )
        return sum_pool3d(relu(y), kernel_size=2, stride=2)
    if kind == 'mini_encoder':
        points, point_feats, batch_indices = _point_inputs(inputs)
        y = voxelize(
            points,
            point_feats,
            batch_indices=batch_indices,
        )
        y = relu(subm_conv3d(y, inputs.kernel3_weight, kernel_size=3))
        y = conv3d(y, inputs.kernel3_weight, kernel_size=3, stride=2)
        return avg_pool3d(relu(y), kernel_size=2, stride=2)

    points, point_feats, batch_indices = _point_inputs(inputs)
    quantized = sparse_quantize(
        points,
        batch_indices=batch_indices,
    )
    x = SparseTensor(
        quantized.coords,
        point_feats,
        active_rows=quantized.active_rows,
    )
    return knn_relation(x, k=8)


def _compiled(
    kind: WorkloadKind,
) -> Callable[[WorkloadFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: WorkloadFixture) -> tuple[Any, tuple[Any, ...]]:
        def fn(feats: mx.array) -> mx.array:
            inputs = _compiled_inputs(kind, fixture, feats)
            out = _run(kind, inputs)
            return out.feats

        points = (
            _point_fixture(fixture) if kind in _POINT_WORKLOADS else None
        )
        return fn, (
            points.feats
            if kind in ('voxel_stem', 'mini_encoder')
            else fixture.sparse.feats,
        )

    return factory


def _backward(
    kind: WorkloadKind,
) -> Callable[[WorkloadFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: WorkloadFixture) -> tuple[Any, tuple[Any, ...]]:
        def loss(feats: mx.array) -> mx.array:
            inputs = _compiled_inputs(kind, fixture, feats)
            return mx.sum(_run(kind, inputs).feats)

        points = (
            _point_fixture(fixture) if kind in _POINT_WORKLOADS else None
        )
        args = (
            points.feats
            if kind in ('voxel_stem', 'mini_encoder')
            else fixture.sparse.feats
        )
        return mx.grad(loss), (args,)

    return factory


def _compiled_inputs(
    kind: WorkloadKind,
    fixture: WorkloadFixture,
    feats: mx.array,
) -> WorkloadInputs:
    point_workload = kind in ('voxel_stem', 'mini_encoder')
    points = None if fixture.points is None else fixture.points.points
    point_feats = None if fixture.points is None else fixture.points.feats
    batch_indices = (
        None if fixture.points is None else fixture.points.batch_indices
    )
    return WorkloadInputs(
        x=SparseTensor(
            fixture.sparse.coords,
            feats if not point_workload else fixture.sparse.feats,
            batch_counts=fixture.sparse.batch_counts,
        ),
        points=points,
        point_feats=feats if point_workload else point_feats,
        batch_indices=batch_indices,
        linear_weight=fixture.linear_weight,
        bias=fixture.bias,
        kernel3_weight=fixture.kernel3_weight,
    )


_POINT_WORKLOADS = ('voxel_stem', 'mini_encoder', 'neighbor_pipeline')


def _point_fixture(fixture: WorkloadFixture) -> PointArrays:
    if fixture.points is None:
        raise ValueError('point fixture is required for this workload.')
    return fixture.points


def _point_inputs(
    inputs: WorkloadInputs,
) -> tuple[mx.array, mx.array, mx.array]:
    if (
        inputs.points is None
        or inputs.point_feats is None
        or inputs.batch_indices is None
    ):
        raise ValueError('point inputs are required for this workload.')
    return inputs.points, inputs.point_feats, inputs.batch_indices
