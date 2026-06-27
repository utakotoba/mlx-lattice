from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from mlx_lattice.core import SparseTensor
from mlx_lattice.ops import cat, crop, prune_mask, sparse_add

from mlx_lattice_bench.cases.common import benchmark_n, param_grid
from mlx_lattice_bench.datasets import SparseArrays, sparse_arrays
from mlx_lattice_bench.harness import BenchmarkCase


@dataclass(frozen=True, slots=True)
class TensorInputs:
    x: SparseTensor
    y: SparseTensor
    mask: mx.array
    crop_max_x: int


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
    return (
        BenchmarkCase(
            name='prune_mask',
            group='tensor',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: prune_mask(inputs.x, inputs.mask),
            units=('elements', 'n_in'),
        ),
        BenchmarkCase(
            name='sparse_add_outer',
            group='tensor',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: sparse_add(inputs.x, inputs.y),
            backward=_backward_sparse_add,
            units=('elements', 'n_in'),
        ),
        BenchmarkCase(
            name='sparse_cat_outer',
            group='tensor',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: cat([inputs.x, inputs.y], join='outer'),
            units=('elements', 'n_in'),
        ),
        BenchmarkCase(
            name='sparse_crop',
            group='tensor',
            params=params,
            setup=_setup,
            prepare=_prepare,
            run=lambda inputs: crop(
                inputs.x,
                min_coord=(0, 0, 0),
                max_coord=(inputs.crop_max_x, inputs.x.capacity, 1),
            ),
            units=('elements', 'n_in'),
        ),
    )


def _setup(
    params: Mapping[str, Any],
) -> tuple[SparseArrays, SparseTensor, mx.array]:
    rows = benchmark_n(params)
    arrays = sparse_arrays(rows=rows, channels=int(params['channels']))
    mask = cast(
        mx.array,
        mx.remainder(mx.arange(rows, dtype=mx.int32), 2) == 0,
    )
    shifted = arrays.tensor()
    y_coords = shifted.coords + mx.zeros_like(shifted.coords)
    y_coords = y_coords.at[:, 1].add(1)
    y = SparseTensor(y_coords, shifted.feats * 0.5, stride=shifted.stride)
    return arrays, y, mask


def _prepare(
    fixture: tuple[SparseArrays, SparseTensor, mx.array],
) -> TensorInputs:
    arrays, y, mask = fixture
    x = arrays.tensor()
    return TensorInputs(x, y, mask, x.capacity // 2)


def _backward_sparse_add(
    fixture: tuple[SparseArrays, SparseTensor, mx.array],
):
    x = fixture[0].tensor()
    y = fixture[1]

    def loss(feats: mx.array) -> mx.array:
        return mx.sum(sparse_add(x.replace(feats=feats), y).feats)

    return mx.grad(loss), (x.feats,)
