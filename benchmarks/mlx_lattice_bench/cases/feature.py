from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import (
    batch_norm,
    dropout,
    gelu,
    layer_norm,
    leaky_relu,
    linear,
    relu,
    rms_norm,
    sigmoid,
    silu,
    softplus,
    tanh,
)

from mlx_lattice_bench.cases.common import param_grid
from mlx_lattice_bench.datasets import (
    SparseArrays,
    dense_bias,
    dense_weight,
    sparse_arrays,
)
from mlx_lattice_bench.harness import BenchmarkCase

type FeatureKind = Literal[
    'linear',
    'relu',
    'sigmoid',
    'gelu',
    'silu',
    'leaky_relu',
    'tanh',
    'softplus',
    'dropout',
    'batch_norm',
    'layer_norm',
    'rms_norm',
]


@dataclass(frozen=True, slots=True)
class FeatureFixture:
    arrays: SparseArrays
    weight: mx.array
    bias: mx.array
    channel_weight: mx.array


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    x: SparseTensor
    weight: mx.array
    bias: mx.array
    channel_weight: mx.array


def cases(preset: str) -> tuple[BenchmarkCase, ...]:
    params = tuple(dict(item) for item in param_grid(preset))
    return tuple(
        _case(f'feature_{kind}', kind, params)
        for kind in (
            'linear',
            'relu',
            'sigmoid',
            'gelu',
            'silu',
            'leaky_relu',
            'tanh',
            'softplus',
            'dropout',
            'batch_norm',
            'layer_norm',
            'rms_norm',
        )
    )


def _case(
    name: str,
    kind: FeatureKind,
    params: tuple[Mapping[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        group='feature',
        params=params,
        setup=_setup,
        prepare=_prepare,
        run=lambda inputs: _run(kind, inputs),
        compiled=_compiled(kind),
        backward=_backward(kind),
        units=('elements', 'n_in'),
    )


def _setup(params: Mapping[str, Any]) -> FeatureFixture:
    channels = int(params['channels'])
    return FeatureFixture(
        arrays=sparse_arrays(
            rows=int(params['rows']),
            channels=channels,
            batches=int(params['batches']),
        ),
        weight=dense_weight((channels, channels)),
        bias=dense_bias(channels),
        channel_weight=mx.ones((channels,), dtype=mx.float32),
    )


def _prepare(fixture: FeatureFixture) -> FeatureInputs:
    return FeatureInputs(
        x=fixture.arrays.tensor(),
        weight=fixture.weight,
        bias=fixture.bias,
        channel_weight=fixture.channel_weight,
    )


def _run(kind: FeatureKind, inputs: FeatureInputs) -> SparseTensor:
    x = inputs.x
    if kind == 'linear':
        return linear(x, inputs.weight, inputs.bias)
    if kind == 'relu':
        return relu(x)
    if kind == 'sigmoid':
        return sigmoid(x)
    if kind == 'gelu':
        return gelu(x)
    if kind == 'silu':
        return silu(x)
    if kind == 'leaky_relu':
        return leaky_relu(x)
    if kind == 'tanh':
        return tanh(x)
    if kind == 'softplus':
        return softplus(x)
    if kind == 'dropout':
        return dropout(x, training=False)
    if kind == 'batch_norm':
        return batch_norm(x, weight=inputs.channel_weight, bias=inputs.bias)
    if kind == 'layer_norm':
        return layer_norm(x, weight=inputs.channel_weight, bias=inputs.bias)
    return rms_norm(x, weight=inputs.channel_weight)


def _compiled(
    kind: FeatureKind,
) -> Callable[[FeatureFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: FeatureFixture) -> tuple[Any, tuple[Any, ...]]:
        def fn(feats: mx.array) -> mx.array:
            inputs = FeatureInputs(
                x=SparseTensor(
                    fixture.arrays.coords,
                    feats,
                    batch_counts=fixture.arrays.batch_counts,
                ),
                weight=fixture.weight,
                bias=fixture.bias,
                channel_weight=fixture.channel_weight,
            )
            return _run(kind, inputs).feats

        return fn, (fixture.arrays.feats,)

    return factory


def _backward(
    kind: FeatureKind,
) -> Callable[[FeatureFixture], tuple[Any, tuple[Any, ...]]]:
    def factory(fixture: FeatureFixture) -> tuple[Any, tuple[Any, ...]]:
        def loss(feats: mx.array) -> mx.array:
            inputs = FeatureInputs(
                x=SparseTensor(
                    fixture.arrays.coords,
                    feats,
                    batch_counts=fixture.arrays.batch_counts,
                ),
                weight=fixture.weight,
                bias=fixture.bias,
                channel_weight=fixture.channel_weight,
            )
            return mx.sum(_run(kind, inputs).feats)

        return mx.grad(loss), (fixture.arrays.feats,)

    return factory
