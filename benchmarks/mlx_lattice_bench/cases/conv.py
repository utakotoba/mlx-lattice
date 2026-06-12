from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import (
    SparseArrays,
    dense_weight,
    sparse_arrays,
)
from mlx_lattice_bench.harness import BenchmarkCase

type ConvKind = Literal[
    'pointwise',
    'generic',
    'subm',
    'target_same',
    'target_subset',
    'target_superset',
    'transpose',
    'generative_transpose',
]
type ConvGradTarget = Literal['features', 'weight', 'both']


@dataclass(frozen=True, slots=True)
class ConvFixture:
    arrays: SparseArrays
    dtype: mx.Dtype
    channels_in: int
    channels_out: int
    target_subset_coords: mx.array
    target_superset_coords: mx.array
    pointwise_weight: mx.array
    kernel3_weight: mx.array
    kernel2_weight: mx.array


@dataclass(frozen=True, slots=True)
class ConvInputs:
    x: SparseTensor
    target_same: SparseTensor
    target_subset: SparseTensor
    target_superset: SparseTensor
    transposed: SparseTensor
    pointwise_weight: mx.array
    kernel3_weight: mx.array
    kernel2_weight: mx.array


def cases(
    preset: str,
    *,
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
    channel_pairs: tuple[tuple[int, int], ...] | None = None,
    dtype: str = 'float32',
) -> tuple[BenchmarkCase, ...]:
    params = tuple(
        {**item, 'dtype': dtype}
        for item in _conv_param_grid(
            preset,
            n_values=n_values,
            channels=channels,
            channel_pairs=channel_pairs,
        )
    )
    specs = (
        ('conv3d_pointwise', 'pointwise'),
        ('conv3d_generic', 'generic'),
        ('subm_conv3d', 'subm'),
        ('conv3d_target_same', 'target_same'),
        ('conv3d_target_subset', 'target_subset'),
        ('conv3d_target_superset', 'target_superset'),
        ('conv_transpose3d', 'transpose'),
        ('generative_conv_transpose3d', 'generative_transpose'),
    )
    forward_cases = tuple(_case(name, kind, params) for name, kind in specs)
    backward_cases = tuple(
        _backward_case(f'{name}_{suffix}', kind, target, params)
        for name, kind in specs
        for suffix, target in (
            ('dfeatures', 'features'),
            ('dweight', 'weight'),
        )
    )
    return forward_cases + backward_cases


def _case(
    name: str,
    kind: ConvKind,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='conv',
        params=params,
        setup=_setup,
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs),
        compiled=_compiled(kind),
        backward=_backward(kind, 'both'),
        units=('elements', 'n_in', 'n_out'),
    )


def _backward_case(
    name: str,
    kind: ConvKind,
    target: ConvGradTarget,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='conv',
        params=params,
        setup=_setup,
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs),
        backward=_backward(kind, target),
        units=('elements', 'n_in', 'n_out'),
        modes=('backward',),
    )


def _setup(params: Mapping[str, Any]) -> ConvFixture:
    channels_in = int(params.get('channels_in', params['channels']))
    channels_out = int(params.get('channels_out', params['channels']))
    dtype = _dtype(params)
    arrays = sparse_arrays(
        rows=benchmark_n(params),
        channels=channels_in,
        batches=int(params['batches']),
        dtype=dtype,
    )
    superset = sparse_arrays(
        rows=benchmark_n(params) + max(1, benchmark_n(params) // 4),
        channels=channels_in,
        batches=int(params['batches']),
        dtype=dtype,
    )
    return ConvFixture(
        arrays=arrays,
        dtype=dtype,
        channels_in=channels_in,
        channels_out=channels_out,
        target_subset_coords=arrays.coords[::2],
        target_superset_coords=superset.coords,
        pointwise_weight=dense_weight(
            (channels_out, 1, 1, 1, channels_in), dtype=dtype
        ),
        kernel3_weight=dense_weight(
            (channels_out, 3, 3, 3, channels_in), dtype=dtype
        ),
        kernel2_weight=dense_weight(
            (channels_out, 2, 2, 2, channels_in), dtype=dtype
        ),
    )


def _prepare(fixture: ConvFixture) -> ConvInputs:
    x = fixture.arrays.tensor()
    return ConvInputs(
        x=x,
        target_same=SparseTensor(
            fixture.arrays.coords,
            mx.zeros_like(fixture.arrays.feats),
            coord_manager=x.coord_manager,
        ),
        target_subset=SparseTensor(
            fixture.target_subset_coords,
            mx.zeros(
                (
                    fixture.target_subset_coords.shape[0],
                    fixture.channels_out,
                ),
                dtype=x.dtype,
            ),
            coord_manager=x.coord_manager,
        ),
        target_superset=SparseTensor(
            fixture.target_superset_coords,
            mx.zeros(
                (
                    fixture.target_superset_coords.shape[0],
                    fixture.channels_out,
                ),
                dtype=x.dtype,
            ),
            coord_manager=x.coord_manager,
        ),
        transposed=fixture.arrays.tensor(stride=2),
        pointwise_weight=fixture.pointwise_weight,
        kernel3_weight=fixture.kernel3_weight,
        kernel2_weight=fixture.kernel2_weight,
    )


def _run(kind: ConvKind, inputs: ConvInputs) -> SparseTensor:
    if kind == 'pointwise':
        return conv3d(inputs.x, inputs.pointwise_weight, kernel_size=1)
    if kind == 'generic':
        return conv3d(inputs.x, inputs.kernel3_weight, kernel_size=3)
    if kind == 'subm':
        return subm_conv3d(inputs.x, inputs.kernel3_weight, kernel_size=3)
    if kind == 'target_same':
        return conv3d(
            inputs.x,
            inputs.kernel3_weight,
            kernel_size=3,
            coordinates=inputs.target_same,
        )
    if kind == 'target_subset':
        return conv3d(
            inputs.x,
            inputs.kernel3_weight,
            kernel_size=3,
            coordinates=inputs.target_subset,
        )
    if kind == 'target_superset':
        return conv3d(
            inputs.x,
            inputs.kernel3_weight,
            kernel_size=3,
            coordinates=inputs.target_superset,
        )
    if kind == 'transpose':
        return conv_transpose3d(
            inputs.transposed,
            inputs.kernel2_weight,
            kernel_size=2,
            stride=2,
        )
    return generative_conv_transpose3d(
        inputs.transposed,
        inputs.kernel2_weight,
        kernel_size=2,
        stride=2,
    )


def _compiled(
    kind: ConvKind,
) -> Callable[[ConvFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: ConvFixture) -> tuple[Any, tuple[Any, ...]]:
        weight = _weight_for(kind, fixture)
        stride = 2 if kind in ('transpose', 'generative_transpose') else 1

        def fn(feats: mx.array, weight_arg: mx.array) -> mx.array:
            x = SparseTensor(
                fixture.arrays.coords,
                feats,
                stride=stride,
                batch_counts=fixture.arrays.batch_counts,
            )
            return _run(
                kind, _compiled_inputs(kind, x, weight_arg, fixture)
            ).feats

        return fn, (fixture.arrays.feats, weight)

    return factory


def _backward(
    kind: ConvKind,
    target: ConvGradTarget,
) -> Callable[[ConvFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: ConvFixture) -> tuple[Any, tuple[Any, ...]]:
        weight = _weight_for(kind, fixture)
        stride = 2 if kind in ('transpose', 'generative_transpose') else 1
        base = fixture.arrays.tensor(stride=stride)

        def loss(feats: mx.array, weight_arg: mx.array) -> mx.array:
            x = base.replace(feats=feats)
            return mx.sum(
                _run(
                    kind, _compiled_inputs(kind, x, weight_arg, fixture)
                ).feats
            )

        return mx.grad(loss, argnums=_argnums_for(target)), (
            fixture.arrays.feats,
            weight,
        )

    return factory


def _argnums_for(target: ConvGradTarget) -> int | tuple[int, int]:
    if target == 'features':
        return 0
    if target == 'weight':
        return 1
    return (0, 1)


def _compiled_inputs(
    kind: ConvKind,
    x: SparseTensor,
    weight: mx.array,
    fixture: ConvFixture | None = None,
) -> ConvInputs:
    empty = mx.array([], dtype=x.dtype)
    target_same = _target_tensor(x, x.coords)
    target_subset = (
        target_same
        if fixture is None
        else _target_tensor(x, fixture.target_subset_coords)
    )
    target_superset = (
        target_same
        if fixture is None
        else _target_tensor(x, fixture.target_superset_coords)
    )
    return ConvInputs(
        x=x,
        target_same=target_same,
        target_subset=target_subset,
        target_superset=target_superset,
        transposed=x,
        pointwise_weight=weight if kind == 'pointwise' else empty,
        kernel3_weight=weight
        if kind
        in (
            'generic',
            'subm',
            'target_same',
            'target_subset',
            'target_superset',
        )
        else empty,
        kernel2_weight=weight
        if kind in ('transpose', 'generative_transpose')
        else empty,
    )


def _weight_for(kind: ConvKind, fixture: ConvFixture) -> mx.array:
    if kind == 'pointwise':
        return fixture.pointwise_weight
    if kind in (
        'generic',
        'subm',
        'target_same',
        'target_subset',
        'target_superset',
    ):
        return fixture.kernel3_weight
    return fixture.kernel2_weight


def _target_tensor(x: SparseTensor, coords: mx.array) -> SparseTensor:
    return SparseTensor(
        coords,
        mx.zeros((coords.shape[0], x.channels), dtype=x.dtype),
        coord_manager=x.coord_manager,
    )


def _dtype(params: Mapping[str, Any]) -> mx.Dtype:
    name = str(params.get('dtype', 'float32'))
    if name == 'float32':
        return mx.float32
    if name == 'float16':
        return mx.float16
    raise ValueError("dtype must be 'float32' or 'float16'.")


def _conv_param_grid(
    preset: str,
    *,
    n_values: tuple[int, ...] | None,
    channels: tuple[int, ...] | None,
    channel_pairs: tuple[tuple[int, int], ...] | None,
) -> tuple[Mapping[str, Any], ...]:
    if channel_pairs is None:
        return param_grid(
            preset,
            n_values=n_values,
            channels=(16,) if channels is None else channels,
        )

    base = param_grid(
        preset,
        n_values=n_values,
        channels=(1,),
    )
    params = []
    for item in base:
        for channels_in, channels_out in channel_pairs:
            copied = dict(item)
            copied['channels'] = channels_in
            copied['channels_in'] = channels_in
            copied['channels_out'] = channels_out
            params.append(copied)
    return tuple(params)
