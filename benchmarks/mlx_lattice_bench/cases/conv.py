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
    'transpose',
    'generative_transpose',
]


@dataclass(frozen=True, slots=True)
class ConvFixture:
    arrays: SparseArrays
    pointwise_weight: mx.array
    kernel3_weight: mx.array
    kernel2_weight: mx.array


@dataclass(frozen=True, slots=True)
class ConvInputs:
    x: SparseTensor
    transposed: SparseTensor
    pointwise_weight: mx.array
    kernel3_weight: mx.array
    kernel2_weight: mx.array


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
            ('conv3d_pointwise', 'pointwise'),
            ('conv3d_generic', 'generic'),
            ('subm_conv3d', 'subm'),
            ('conv_transpose3d', 'transpose'),
            ('generative_conv_transpose3d', 'generative_transpose'),
        )
    )


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
        backward=_backward(kind),
        units=('elements', 'n_in', 'n_out'),
    )


def _setup(params: Mapping[str, Any]) -> ConvFixture:
    channels = int(params['channels'])
    arrays = sparse_arrays(
        rows=benchmark_n(params),
        channels=channels,
        batches=int(params['batches']),
    )
    return ConvFixture(
        arrays=arrays,
        pointwise_weight=dense_weight((channels, 1, 1, 1, channels)),
        kernel3_weight=dense_weight((channels, 3, 3, 3, channels)),
        kernel2_weight=dense_weight((channels, 2, 2, 2, channels)),
    )


def _prepare(fixture: ConvFixture) -> ConvInputs:
    return ConvInputs(
        x=fixture.arrays.tensor(),
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
            return _run(kind, _compiled_inputs(kind, x, weight_arg)).feats

        return fn, (fixture.arrays.feats, weight)

    return factory


def _backward(
    kind: ConvKind,
) -> Callable[[ConvFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: ConvFixture) -> tuple[Any, tuple[Any, ...]]:
        weight = _weight_for(kind, fixture)
        stride = 2 if kind in ('transpose', 'generative_transpose') else 1

        def loss(feats: mx.array, weight_arg: mx.array) -> mx.array:
            x = SparseTensor(
                fixture.arrays.coords,
                feats,
                stride=stride,
                batch_counts=fixture.arrays.batch_counts,
            )
            return mx.sum(
                _run(kind, _compiled_inputs(kind, x, weight_arg)).feats
            )

        return mx.grad(loss, argnums=(0, 1)), (fixture.arrays.feats, weight)

    return factory


def _compiled_inputs(
    kind: ConvKind,
    x: SparseTensor,
    weight: mx.array,
) -> ConvInputs:
    empty = mx.array([], dtype=mx.float32)
    return ConvInputs(
        x=x,
        transposed=x,
        pointwise_weight=weight if kind == 'pointwise' else empty,
        kernel3_weight=weight if kind in ('generic', 'subm') else empty,
        kernel2_weight=weight
        if kind in ('transpose', 'generative_transpose')
        else empty,
    )


def _weight_for(kind: ConvKind, fixture: ConvFixture) -> mx.array:
    if kind == 'pointwise':
        return fixture.pointwise_weight
    if kind in ('generic', 'subm'):
        return fixture.kernel3_weight
    return fixture.kernel2_weight
