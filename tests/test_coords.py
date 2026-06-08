from __future__ import annotations

from typing import Any, cast

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import backend_info  # noqa: E402
from mlx_lattice.core import (  # noqa: E402
    CoordinateManager,
    SparseTensor,
    build_generative_map,
    build_kernel_map,
    build_transposed_kernel_map,
    downsample_coords,
    intersection_coords,
    kernel_offsets,
    lookup_coords,
    union_coords,
)


def test_kernel_offsets_match_centered_and_even_conventions() -> None:
    assert kernel_offsets(1) == ((0, 0, 0),)
    assert kernel_offsets((3, 1, 1)) == (
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
    )
    assert kernel_offsets((2, 1, 1), dilation=(2, 1, 1)) == (
        (0, 0, 0),
        (2, 0, 0),
    )


def test_coordinate_set_ops_preserve_first_seen_order() -> None:
    lhs = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    rhs = mx.array(
        [[0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )

    assert downsample_coords(rhs, stride=2).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    assert union_coords(lhs, rhs).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert intersection_coords(lhs, rhs).tolist() == [[0, 1, 0, 0]]
    assert lookup_coords(lhs, rhs).tolist() == [1, -1]


def test_build_kernel_map_emits_compact_edge_contract() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )

    mapping = build_kernel_map(coords, kernel_size=(3, 1, 1))

    assert mapping.out_coords is not None
    assert mapping.out_coords.tolist() == coords.tolist()
    assert mapping.kernel_offsets == (
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
    )
    assert mapping.in_rows.tolist() == [0, 1, 0, 1, 2, 1, 2]
    assert mapping.out_rows.tolist() == [1, 2, 0, 1, 2, 0, 1]
    assert mapping.kernel_ids.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_build_strided_kernel_map_downsamples_output_coords() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )

    mapping = build_kernel_map(coords, kernel_size=1, stride=2)

    assert mapping.out_coords is not None
    assert mapping.out_coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert mapping.in_rows.tolist() == [0, 2]
    assert mapping.out_rows.tolist() == [0, 1]
    assert mapping.kernel_ids.tolist() == [0, 0]


def test_generative_and_transposed_maps_cover_output_policy() -> None:
    coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)

    generated = build_generative_map(
        coords,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    transposed = build_transposed_kernel_map(
        coords,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )

    assert generated.out_coords is not None
    assert generated.out_coords.tolist() == [
        [0, 2, 0, 0],
        [0, 3, 0, 0],
    ]
    assert generated.in_rows.tolist() == [0, 0]
    assert generated.out_rows.tolist() == [0, 1]
    assert generated.kernel_ids.tolist() == [0, 1]
    assert transposed.out_coords is not None
    assert transposed.out_coords.tolist() == generated.out_coords.tolist()


def test_generative_map_runs_with_gpu_default_when_metal_is_available() -> (
    None
):
    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    if not capabilities['metal']:
        pytest.skip('Metal backend was not built')
    if not hasattr(mx, 'metal') or not mx.metal.is_available():
        pytest.skip('Metal device is not available')

    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mapping = build_generative_map(
            mx.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=mx.int32),
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        mx.eval(
            mapping.out_coords,
            mapping.in_rows,
            mapping.out_rows,
            mapping.kernel_ids,
        )
    finally:
        mx.set_default_device(previous_device)

    assert mapping.out_coords is not None
    assert mapping.out_coords.shape == (4, 4)
    assert mapping.in_rows.tolist() == [0, 0, 1, 1]
    assert mapping.out_rows.tolist() == [0, 1, 2, 3]
    assert mapping.kernel_ids.tolist() == [0, 1, 0, 1]


def test_coordinate_primitives_run_with_gpu_default_when_metal_is_available() -> (
    None
):
    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    if not capabilities['metal']:
        pytest.skip('Metal backend was not built')
    if not hasattr(mx, 'metal') or not mx.metal.is_available():
        pytest.skip('Metal device is not available')

    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        lhs = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
            dtype=mx.int32,
        )
        rhs = mx.array(
            [[0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        downsampled = downsample_coords(rhs, stride=2)
        unioned = union_coords(lhs, rhs)
        intersected = intersection_coords(lhs, rhs)
        looked_up = lookup_coords(lhs, rhs)

        coords = mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        )
        forward = build_kernel_map(coords, kernel_size=(3, 1, 1))
        transposed = build_transposed_kernel_map(
            mx.array([[0, 1, 0, 0]], dtype=mx.int32),
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        mx.eval(
            downsampled,
            unioned,
            intersected,
            looked_up,
            forward.in_rows,
            forward.out_rows,
            forward.kernel_ids,
            transposed.out_coords,
            transposed.in_rows,
            transposed.out_rows,
            transposed.kernel_ids,
        )
    finally:
        mx.set_default_device(previous_device)

    assert downsampled.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert unioned.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
    ]
    assert intersected.tolist() == [[0, 1, 0, 0]]
    assert looked_up.tolist() == [1, -1]
    assert forward.in_rows.tolist() == [0, 1, 0, 1, 2, 1, 2]
    assert forward.out_rows.tolist() == [1, 2, 0, 1, 2, 0, 1]
    assert forward.kernel_ids.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert transposed.out_coords is not None
    assert transposed.out_coords.tolist() == [
        [0, 2, 0, 0],
        [0, 3, 0, 0],
    ]
    assert transposed.in_rows.tolist() == [0, 0]
    assert transposed.out_rows.tolist() == [0, 1]
    assert transposed.kernel_ids.tolist() == [0, 1]


def test_coordinate_manager_caches_kernel_maps() -> None:
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    manager = CoordinateManager()
    key = manager.insert_coords(coords)

    first = manager.kernel_map(key, kernel_size=(3, 1, 1))
    second = manager.kernel_map(key, kernel_size=(3, 1, 1))
    tensor_map = SparseTensor(coords, mx.ones((2, 1))).kernel_map(
        kernel_size=(3, 1, 1)
    )

    assert first is second
    assert tensor_map.kernel_offsets == first.kernel_offsets
