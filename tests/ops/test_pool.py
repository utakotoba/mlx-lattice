from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    avg_pool3d,
    global_avg_pool,
    global_max_pool,
    global_sum_pool,
    max_pool3d,
    pool3d,
    sparse_collate,
    sum_pool3d,
)
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    mx,
)

pytestmark = [
    pytest.mark.ops,
    pytest.mark.pool,
    pytest.mark.usefixtures('selected_backend'),
]


def test_local_pooling_uses_fused_native_neighborhood_reductions() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=mx.float32,
    )
    x = SparseTensor(coords, feats)

    summed = sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)
    maxed = max_pool3d(x, kernel_size=(3, 1, 1), stride=1)
    averaged = avg_pool3d(x, kernel_size=(3, 1, 1), stride=1)

    assert active_coords(summed) == coords.tolist()
    assert active_feats(summed).tolist() == [
        [3.0, 30.0],
        [6.0, 60.0],
        [5.0, 50.0],
    ]
    assert active_feats(maxed).tolist() == [
        [2.0, 20.0],
        [3.0, 30.0],
        [3.0, 30.0],
    ]
    assert active_feats(averaged).tolist() == [
        [1.5, 15.0],
        [2.0, 20.0],
        [2.5, 25.0],
    ]


def test_local_pooling_modes_are_autogradable() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)

    def sum_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    def avg_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    def max_loss(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats)

    assert mx.grad(sum_loss)(feats).tolist() == [[2.0], [3.0], [2.0]]
    assert_nested_close(
        mx.grad(avg_loss)(feats).tolist(),
        [[0.8333333730697632], [1.3333333730697632], [0.8333333730697632]],
    )
    assert mx.grad(max_loss)(feats).tolist() == [[0.0], [1.0], [2.0]]


def test_max_pool3d_tie_policy_matches_mlx_transform_contract() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[2.0], [2.0], [1.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)

    def pooled(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    def loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(pooled(feats_arg))

    outputs, grads = mx.vjp(
        pooled,
        [feats],
        [mx.ones((3, 1), dtype=mx.float32)],
    )
    _, jvps = mx.jvp(pooled, [feats], [tangent])

    assert outputs[0].tolist() == [[2.0], [2.0], [2.0]]
    assert mx.grad(loss)(feats).tolist() == [[1.0], [2.0], [0.0]]
    assert grads[0].tolist() == [[1.0], [2.0], [0.0]]
    assert jvps[0].tolist() == [[10.0], [10.0], [20.0]]


def test_pooling_modes_are_compatible_with_mx_compile(
    compile_backend,
) -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)

    def summed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    def averaged(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    def maxed(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

    assert mx.compile(summed)(feats).tolist() == [[3.0], [6.0], [5.0]]
    assert mx.compile(averaged)(feats).tolist() == [[1.5], [2.0], [2.5]]
    assert mx.compile(maxed)(feats).tolist() == [[2.0], [3.0], [3.0]]


def test_strided_pooling_updates_output_stride_and_manager_context() -> (
    None
):
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
    x = SparseTensor(coords, feats)

    out = sum_pool3d(x, kernel_size=1, stride=2)

    assert active_coords(out) == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert active_feats(out).tolist() == [[1.0], [3.0]]
    assert out.stride == (2, 2, 2)
    assert out.coord_manager is x.coord_manager
    assert out.coord_key != x.coord_key
    assert out.coord_manager.owns(out.coord_key)


def test_global_pooling_reduces_each_batch_independently() -> None:
    x = sparse_collate(
        [
            mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32),
            mx.array([[2, 0, 0], [3, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([[1.0, 10.0], [2.0, 20.0]], dtype=mx.float32),
            mx.array([[3.0, 30.0], [5.0, 50.0]], dtype=mx.float32),
        ],
    )

    assert global_sum_pool(x).tolist() == [[3.0, 30.0], [8.0, 80.0]]
    assert global_avg_pool(x).tolist() == [[1.5, 15.0], [4.0, 40.0]]
    assert global_max_pool(x).tolist() == [[2.0, 20.0], [5.0, 50.0]]


def test_global_pooling_uses_explicit_empty_batch_metadata() -> None:
    x = sparse_collate(
        [
            mx.array([], dtype=mx.int32).reshape((0, 3)),
            mx.array([[2, 0, 0]], dtype=mx.int32),
        ],
        [
            mx.array([], dtype=mx.float32).reshape((0, 2)),
            mx.array([[3.0, 30.0]], dtype=mx.float32),
        ],
    )

    assert global_sum_pool(x).tolist() == [[0.0, 0.0], [3.0, 30.0]]
    assert global_avg_pool(x).tolist() == [[0.0, 0.0], [3.0, 30.0]]
    with pytest.raises(ValueError, match='empty batches'):
        global_max_pool(x)


def test_pool3d_rejects_invalid_mode_and_dtype() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0]], dtype=mx.int32),
        mx.ones((1, 1), dtype=mx.float32),
    )
    with pytest.raises(ValueError, match='mode'):
        pool3d(x, mode=cast('Any', 'median'))

    half = x.astype(mx.float16)
    with pytest.raises(ValueError, match='float32'):
        sum_pool3d(half)
