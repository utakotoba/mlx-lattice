from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

mx = pytest.importorskip('mlx.core')


def assert_nested_close(
    actual: object,
    expected: object,
    *,
    abs: float = 1e-6,
) -> None:
    if isinstance(actual, list | tuple) and isinstance(
        expected, list | tuple
    ):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(
            actual, expected, strict=True
        ):
            assert_nested_close(actual_item, expected_item, abs=abs)
        return
    assert actual == pytest.approx(expected, abs=abs)


def assert_same_sparse_identity(left: Any, right: Any) -> None:
    assert left.coord_key == right.coord_key
    assert left.coord_manager is right.coord_manager
    assert left.coords is right.coords


def active_count(tensor: Any) -> int:
    return int(tensor.active_rows.tolist()[0])


def active_coords(tensor: Any) -> list[list[int]]:
    return cast(
        'list[list[int]]',
        tensor.coords[: active_count(tensor)].tolist(),
    )


def active_feats(tensor: Any) -> Any:
    return tensor.feats[: active_count(tensor)]


def skip_without_metal() -> None:
    from mlx_lattice import backend_info

    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    if not capabilities['metal']:
        pytest.skip('Metal backend was not built')
    if not hasattr(mx, 'metal') or not mx.metal.is_available():
        pytest.skip('Metal device is not available')


def run_with_gpu_default[T](fn: Callable[[], T]) -> T:
    skip_without_metal()
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        return fn()
    finally:
        mx.set_default_device(previous)


def line_tensor() -> Any:
    from mlx_lattice import SparseTensor

    return SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )
