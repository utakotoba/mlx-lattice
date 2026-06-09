from __future__ import annotations

from typing import cast

import pytest

from mlx_lattice.ops import sparse_quantize, voxelize
from tests.support import (
    active_coords,
    active_feats,
    assert_nested_close,
    mx,
    run_with_gpu_default,
)


def _active_rows(values: mx.array, count: mx.array) -> list[int]:
    return cast('list[int]', values[: int(count.tolist()[0])].tolist())


def _active_coords(values: mx.array, count: mx.array) -> list[list[int]]:
    return cast(
        'list[list[int]]', values[: int(count.tolist()[0])].tolist()
    )


def test_sparse_quantize_defines_ordered_voxel_metadata_contract() -> None:
    points = mx.array(
        [
            [0.2, 0.2, 0.2],
            [0.8, 0.1, 0.1],
            [1.2, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 2.1, 0.0],
        ],
        dtype=mx.float32,
    )
    batches = mx.array([0, 0, 0, 0, 1], dtype=mx.int32)

    quantized = sparse_quantize(
        points,
        voxel_size=(1.0, 1.0, 1.0),
        batch_indices=batches,
    )

    assert quantized.capacity == 5
    assert quantized.active_count is quantized.active_rows
    assert quantized.active_rows.tolist() == [4]
    assert _active_coords(quantized.coords, quantized.active_count) == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 2, 0],
    ]
    assert quantized.inverse_rows.tolist() == [0, 0, 1, 2, 3]
    assert _active_rows(quantized.counts, quantized.active_count) == [
        2,
        1,
        1,
        1,
    ]


def test_sparse_quantize_respects_origin_and_active_rows() -> None:
    points = mx.array(
        [
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [99.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    active_rows = mx.array([3], dtype=mx.int32)

    quantized = sparse_quantize(
        points,
        voxel_size=(2.0, 1.0, 1.0),
        origin=(0.5, 0.0, 0.0),
        active_rows=active_rows,
    )

    assert quantized.active_rows.tolist() == [2]
    assert _active_coords(quantized.coords, quantized.active_count) == [
        [0, -1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert quantized.inverse_rows.tolist() == [0, 1, 1, -1]
    assert quantized.counts.tolist() == [1, 2, 0, 0]


def test_voxelize_aggregates_features_into_sparse_tensor() -> None:
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array(
        [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
        dtype=mx.float32,
    )

    mean = voxelize(points, feats, voxel_size=1.0, reduction='mean')
    summed = voxelize(points, feats, voxel_size=1.0, reduction='sum')

    assert active_coords(mean) == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert_nested_close(
        active_feats(mean).tolist(), [[2.0, 20.0], [5.0, 50.0]]
    )
    assert_nested_close(
        active_feats(summed).tolist(), [[4.0, 40.0], [5.0, 50.0]]
    )
    assert mean.stride == (1, 1, 1)


def test_voxelize_feature_aggregation_is_autogradable() -> None:
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)
    tangent = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)

    def features(feats_arg: mx.array) -> mx.array:
        return voxelize(points, feats_arg, reduction='mean').feats

    def loss(feats_arg: mx.array) -> mx.array:
        return mx.sum(features(feats_arg))

    grads = mx.grad(loss)(feats)
    _, jvps = mx.jvp(features, [feats], [tangent])
    jvp_values = jvps[0] if isinstance(jvps, list) else jvps
    compiled = mx.compile(features)(feats)

    assert_nested_close(grads.tolist(), [[0.5], [0.5], [1.0]])
    assert_nested_close(jvp_values.tolist()[:2], [[15.0], [30.0]])
    assert_nested_close(compiled.tolist()[:2], [[2.0], [5.0]])


def test_quantization_ops_reject_ambiguous_contracts() -> None:
    points = mx.ones((2, 3), dtype=mx.float32)
    feats = mx.ones((2, 1), dtype=mx.float32)

    with pytest.raises(ValueError, match='points must have shape'):
        sparse_quantize(mx.ones((2, 4), dtype=mx.float32))
    with pytest.raises(ValueError, match='points must be float32'):
        sparse_quantize(mx.ones((2, 3), dtype=mx.int32))
    with pytest.raises(ValueError, match='batch_indices'):
        sparse_quantize(points, batch_indices=mx.ones((2,), dtype=mx.int64))
    with pytest.raises(ValueError, match='voxel_size'):
        sparse_quantize(points, voxel_size=0.0)
    with pytest.raises(ValueError, match='reduction'):
        voxelize(points, feats, reduction=cast('object', 'max'))
    with pytest.raises(ValueError, match='matching rows'):
        voxelize(points, mx.ones((3, 1), dtype=mx.float32))


def test_metal_quantization_and_voxelization_match_cpu_contract_when_available() -> (
    None
):
    def run() -> tuple[
        list[list[int]],
        list[int],
        list[int],
        list[list[float]],
        list[list[float]],
        list[int],
    ]:
        points = mx.array(
            [
                [0.2, 0.2, 0.2],
                [0.8, 0.1, 0.1],
                [1.2, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.0, 2.1, 0.0],
            ],
            dtype=mx.float32,
        )
        feats = mx.array(
            [[1.0], [3.0], [5.0], [7.0], [11.0]],
            dtype=mx.float32,
        )
        batches = mx.array([0, 0, 0, 0, 1], dtype=mx.int32)
        quantized = sparse_quantize(points, batch_indices=batches)
        mean = voxelize(
            points,
            feats,
            batch_indices=batches,
            reduction='mean',
        )
        summed = voxelize(
            points,
            feats,
            batch_indices=batches,
            reduction='sum',
        )
        mx.eval(
            quantized.coords,
            quantized.active_rows,
            quantized.inverse_rows,
            quantized.counts,
            mean.feats,
            summed.feats,
            mean.active_rows,
        )
        return (
            _active_coords(quantized.coords, quantized.active_count),
            cast('list[int]', quantized.inverse_rows.tolist()),
            _active_rows(quantized.counts, quantized.active_count),
            active_feats(mean).tolist(),
            active_feats(summed).tolist(),
            cast('list[int]', mean.active_rows.tolist()),
        )

    (
        coords,
        inverse_rows,
        counts,
        mean_feats,
        sum_feats,
        active_rows,
    ) = run_with_gpu_default(run)

    assert coords == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 2, 0],
    ]
    assert inverse_rows == [0, 0, 1, 2, 3]
    assert counts == [2, 1, 1, 1]
    assert_nested_close(mean_feats, [[2.0], [5.0], [7.0], [11.0]])
    assert_nested_close(sum_feats, [[4.0], [5.0], [7.0], [11.0]])
    assert active_rows == [4]
