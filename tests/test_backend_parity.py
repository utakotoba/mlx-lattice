from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import (  # noqa: E402
    SparseTensor,
    build_generative_map,
    build_kernel_map,
    cat,
    contains_coords,
    conv3d,
    downsample,
    generative_conv_transpose3d,
    intersection_coords,
    lookup_coords,
    pool3d,
    prune,
    relu,
    sigmoid,
    topk_rows,
    union_coords,
)


def test_cpu_and_available_gpu_backends_match_sparse_surface():
    if not _gpu_available():
        pytest.skip('No MLX GPU backend is available.')

    cpu = _run_surface(mx.cpu, mx.int64)
    gpu = _run_surface(mx.gpu, mx.int32)

    assert gpu.keys() == cpu.keys()
    for key in cpu:
        assert _nested_close(gpu[key], cpu[key]), key


def test_available_gpu_backend_is_reproducible():
    if not _gpu_available():
        pytest.skip('No MLX GPU backend is available.')

    first = _run_surface(mx.gpu, mx.int32)
    second = _run_surface(mx.gpu, mx.int32)

    assert first.keys() == second.keys()
    for key in first:
        assert _nested_close(first[key], second[key]), key


def _run_surface(
    device: mx.DeviceType, coord_dtype: mx.Dtype
) -> dict[str, Any]:
    return _with_default_device(
        mx.Device(device), lambda: _collect_surface(coord_dtype)
    )


def _collect_surface(coord_dtype: mx.Dtype) -> dict[str, Any]:
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=coord_dtype,
    )
    feats = mx.array(
        [
            [1.0, -1.0],
            [2.0, 0.5],
            [3.0, 1.0],
            [-1.0, 2.0],
            [0.5, -0.5],
            [1.5, 2.5],
        ],
        dtype=mx.float32,
    )
    x = SparseTensor(coords, feats, batch_counts=(4, 2))
    queries = mx.take(coords, mx.array([1, 4], dtype=mx.int32), axis=0)

    weight = (
        mx.arange(27 * 2 * 4, dtype=mx.float32).reshape((27, 2, 4)) / 17.0
        - 1.5
    )
    strided_weight = (
        mx.arange(8 * 2 * 3, dtype=mx.float32).reshape((8, 2, 3)) / 19.0
        - 0.75
    )
    gen_weight = (
        mx.arange(8 * 2 * 3, dtype=mx.float32).reshape((8, 2, 3)) / 23.0
        + 0.25
    )

    subm_map = build_kernel_map(coords, kernel_size=3, stride=1)
    gen_map = build_generative_map(coords[:2], kernel_size=2, stride=2)
    subm = conv3d(x, weight, kernel_size=3, stride=1)
    strided = conv3d(x, strided_weight, kernel_size=2, stride=2)
    pooled = pool3d(x, kernel_size=2, stride=2)
    generated = generative_conv_transpose3d(
        SparseTensor(coords[:2], feats[:2], stride=2),
        gen_weight,
        kernel_size=2,
        stride=2,
    )
    rows = topk_rows(x, counts=(4, 2), rho=0.5)
    pruned = prune(x, rows)
    merged = cat([relu(x), sigmoid(x)])

    def conv_loss(feats_value: mx.array) -> mx.array:
        return mx.sum(
            conv3d(
                SparseTensor(coords, feats_value, batch_counts=(4, 2)),
                weight,
                kernel_size=3,
            ).feats
        )

    def weight_loss(weight_value: mx.array) -> mx.array:
        return mx.sum(conv3d(x, weight_value, kernel_size=3).feats)

    def gen_loss(feats_value: mx.array) -> mx.array:
        return mx.sum(
            generative_conv_transpose3d(
                SparseTensor(coords[:2], feats_value, stride=2),
                gen_weight,
                kernel_size=2,
                stride=2,
            ).feats
        )

    values = {
        'downsample': downsample(coords, stride=2),
        'coord_union': union_coords(coords[:3], coords[2:5]),
        'coord_intersection': intersection_coords(coords[:4], coords[2:]),
        'coord_lookup': lookup_coords(coords, queries),
        'coord_contains': contains_coords(coords, queries),
        'subm_sizes': subm_map.sizes,
        'subm_valid': mx.sum(subm_map.kernels >= 0),
        'subm_coords': subm.coords,
        'subm_feats': subm.feats,
        'strided_coords': strided.coords,
        'strided_feats': strided.feats,
        'pool_coords': pooled.coords,
        'pool_feats': pooled.feats,
        'gen_map_coords': gen_map.out_coords,
        'gen_coords': generated.coords,
        'gen_feats': generated.feats,
        'topk_rows': rows,
        'pruned_coords': pruned.coords,
        'pruned_feats': pruned.feats,
        'cat_feats': merged.feats,
        'grad_feats': mx.grad(conv_loss)(feats),
        'grad_weight_sum': mx.sum(mx.grad(weight_loss)(weight)),
        'grad_gen_feats': mx.grad(gen_loss)(feats[:2]),
    }
    mx.eval(*values.values())
    return {key: _to_python(value) for key, value in values.items()}


def _with_default_device(device: mx.Device, fn: Callable[[], Any]) -> Any:
    previous = mx.default_device()
    mx.set_default_device(device)
    try:
        return fn()
    finally:
        mx.set_default_device(previous)


def _gpu_available() -> bool:
    return mx.metal.is_available() or mx.cuda.is_available()


def _to_python(value: mx.array) -> Any:
    return value.tolist()


def _nested_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(
            _nested_close(actual_item, expected_item)
            for actual_item, expected_item in zip(
                actual, expected, strict=True
            )
        )
    if isinstance(actual, float) or isinstance(expected, float):
        return actual == pytest.approx(expected, rel=2e-3, abs=2e-3)
    return actual == expected
