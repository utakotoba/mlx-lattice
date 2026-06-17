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
    run_with_gpu_default,
)


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


def test_pooling_modes_are_compatible_with_mx_compile() -> None:
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


def test_metal_local_pooling_matches_cpu_contract_when_available() -> None:
    def run() -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[float]],
    ]:
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
        mx.eval(summed.feats, maxed.feats, averaged.feats)
        return (
            active_feats(summed).tolist(),
            active_feats(maxed).tolist(),
            active_feats(averaged).tolist(),
        )

    assert run_with_gpu_default(run) == (
        [[3.0, 30.0], [6.0, 60.0], [5.0, 50.0]],
        [[2.0, 20.0], [3.0, 30.0], [3.0, 30.0]],
        [[1.5, 15.0], [2.0, 20.0], [2.5, 25.0]],
    )


def test_metal_pooling_gradients_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[float]], list[list[float]], list[list[float]]
    ]:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)

        def sum_loss(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(
                sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats
            )

        def avg_loss(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(
                avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats
            )

        def max_loss(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(
                max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats
            )

        sum_grad = mx.grad(sum_loss)(feats)
        avg_grad = mx.grad(avg_loss)(feats)
        max_grad = mx.grad(max_loss)(feats)
        mx.eval(sum_grad, avg_grad, max_grad)
        return sum_grad.tolist(), avg_grad.tolist(), max_grad.tolist()

    sum_grad, avg_grad, max_grad = run_with_gpu_default(run)
    assert sum_grad == [[2.0], [3.0], [2.0]]
    assert_nested_close(
        avg_grad,
        [[0.8333333730697632], [1.3333333730697632], [0.8333333730697632]],
    )
    assert max_grad == [[0.0], [1.0], [2.0]]


def test_metal_max_pool_tie_policy_matches_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[list[list[float]], list[list[float]]]:
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

        grad = mx.grad(loss)(feats)
        _, jvps = mx.jvp(pooled, [feats], [tangent])
        mx.eval(grad, jvps[0])
        return grad.tolist(), jvps[0].tolist()

    assert run_with_gpu_default(run) == (
        [[1.0], [2.0], [0.0]],
        [[10.0], [10.0], [20.0]],
    )


def test_metal_pooling_jvp_matches_cpu_contract_when_available() -> None:
    def run() -> tuple[
        list[list[float]], list[list[float]], list[list[float]]
    ]:
        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
        tangent = mx.ones_like(feats)

        def summed(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return sum_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

        def averaged(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return avg_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

        def maxed(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return max_pool3d(x, kernel_size=(3, 1, 1), stride=1).feats

        _, sum_jvp = mx.jvp(summed, [feats], [tangent])
        _, avg_jvp = mx.jvp(averaged, [feats], [tangent])
        _, max_jvp = mx.jvp(maxed, [feats], [tangent])
        mx.eval(sum_jvp[0], avg_jvp[0], max_jvp[0])
        return sum_jvp[0].tolist(), avg_jvp[0].tolist(), max_jvp[0].tolist()

    assert run_with_gpu_default(run) == (
        [[2.0], [3.0], [2.0]],
        [[1.0], [1.0], [1.0]],
        [[1.0], [1.0], [1.0]],
    )


def test_metal_strided_pooling_autodiff_uses_forward_topology() -> None:
    def run() -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[float]],
        list[list[float]],
        list[list[float]],
    ]:
        coords = mx.array(
            [[0, row, 0, 0] for row in range(8)],
            dtype=mx.int32,
        )
        feats = mx.array(
            [[float(row)] for row in range(1, 9)],
            dtype=mx.float32,
        )
        tangent = mx.array(
            [[float(row * 10)] for row in range(1, 9)],
            dtype=mx.float32,
        )

        def summed(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return sum_pool3d(x, kernel_size=2, stride=2).feats

        def averaged(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return avg_pool3d(x, kernel_size=2, stride=2).feats

        def maxed(feats_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return max_pool3d(x, kernel_size=2, stride=2).feats

        sum_grad = mx.grad(lambda value: mx.sum(summed(value)))(feats)
        avg_grad = mx.grad(lambda value: mx.sum(averaged(value)))(feats)
        max_grad = mx.grad(lambda value: mx.sum(maxed(value)))(feats)
        _, avg_jvp = mx.jvp(averaged, [feats], [tangent])
        _, max_jvp = mx.jvp(maxed, [feats], [tangent])
        mx.eval(sum_grad, avg_grad, max_grad, avg_jvp[0], max_jvp[0])
        return (
            sum_grad.tolist(),
            avg_grad.tolist(),
            max_grad.tolist(),
            avg_jvp[0][:4].tolist(),
            max_jvp[0][:4].tolist(),
        )

    assert run_with_gpu_default(run) == (
        [[1.0]] * 8,
        [[0.5]] * 8,
        [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
        [[15.0], [35.0], [55.0], [75.0]],
        [[20.0], [40.0], [60.0], [80.0]],
    )


def test_metal_pooling_respects_active_rows_capacity_contract() -> None:
    def run() -> tuple[list[list[int]], list[list[float]], list[int]]:
        coords = mx.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 99, 0, 0],
                [0, 100, 0, 0],
            ],
            dtype=mx.int32,
        )
        feats = mx.array([[1.0], [2.0], [100.0], [200.0]], dtype=mx.float32)
        x = SparseTensor(
            coords,
            feats,
            active_rows=mx.array([2], dtype=mx.int32),
        )

        out = sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)
        mx.eval(out.coords, out.feats, out.active_rows)
        return (
            active_coords(out),
            cast('list[list[float]]', active_feats(out).tolist()),
            cast('list[int]', out.active_rows.tolist()),
        )

    assert run_with_gpu_default(run) == (
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        [[3.0], [3.0]],
        [2],
    )


def test_metal_pooling_rejects_unsupported_coord_dtype() -> None:
    def run() -> None:
        x = SparseTensor(
            mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int64),
            mx.array([[1.0], [2.0]], dtype=mx.float32),
        )
        with pytest.raises(ValueError, match='Metal sparse pooling'):
            sum_pool3d(x, kernel_size=(3, 1, 1), stride=1)

    run_with_gpu_default(run)


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
