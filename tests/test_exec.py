from __future__ import annotations

# ruff: noqa: E402, I001

from typing import Any, cast

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import backend_info
from mlx_lattice.core import KernelMap
from mlx_lattice.ops import build_kernel_map, spmm_edges


def test_spmm_edges_accumulates_repeated_output_rows() -> None:
    mapping = KernelMap(
        mx.array([0, 1, 2, 0], dtype=mx.int32),
        mx.array([0, 0, 1, 1], dtype=mx.int32),
        mx.array([0, 1, 0, 1], dtype=mx.int32),
        kernel_offsets=((0, 0, 0), (1, 0, 0)),
        n_in_rows=3,
        n_out_rows=2,
        n_kernels=2,
    )
    feats = mx.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=mx.float32,
    )
    weights = mx.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ],
        dtype=mx.float32,
    )

    out = spmm_edges(feats, weights, mapping)

    assert out.tolist() == [[11.0, 13.0], [9.0, 11.0]]


def test_spmm_edges_uses_kernel_map_from_coordinate_builder() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    mapping = build_kernel_map(coords, kernel_size=(3, 1, 1))
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    weights = mx.ones((3, 1, 1), dtype=mx.float32)

    out = spmm_edges(feats, weights, mapping)

    assert out.tolist() == [[3.0], [6.0], [5.0]]


def test_spmm_edges_runs_with_gpu_default_when_metal_is_available() -> None:
    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    if not capabilities['metal']:
        pytest.skip('Metal backend was not built')
    if not hasattr(mx, 'metal') or not mx.metal.is_available():
        pytest.skip('Metal device is not available')

    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mapping = KernelMap(
            mx.array([0, 1, 2, 0], dtype=mx.int32),
            mx.array([0, 0, 1, 1], dtype=mx.int32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            kernel_offsets=((0, 0, 0), (1, 0, 0)),
            n_in_rows=3,
            n_out_rows=2,
            n_kernels=2,
        )
        feats = mx.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=mx.float32,
        )
        weights = mx.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ],
            dtype=mx.float32,
        )
        out = spmm_edges(feats, weights, mapping)
        mx.eval(out)
    finally:
        mx.set_default_device(previous_device)

    assert out.tolist() == [[11.0, 13.0], [9.0, 11.0]]
