from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from functools import cache
from types import ModuleType
from typing import Any

import mlx.core as mx

_OPTIONAL_BACKENDS = ('mlx_lattice_cuda13._ext',)


@cache
def _ext() -> ModuleType:
    import mlx.core  # noqa: F401

    for module_name in _OPTIONAL_BACKENDS:
        parent_name = module_name.partition('.')[0]
        if (
            importlib.util.find_spec(parent_name) is not None
            and importlib.util.find_spec(module_name) is not None
        ):
            return importlib.import_module(module_name)

    from . import _ext as native

    return native


def version() -> str:
    return str(_ext().version())


def capabilities() -> Mapping[str, bool]:
    return dict(_ext().capabilities())


def downsample_coords(
    coords: mx.array,
    stride: tuple[int, int, int],
) -> mx.array:
    return _ext().downsample_coords(coords, *stride)


def union_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    return _ext().union_coords(lhs, rhs)


def intersection_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    return _ext().intersection_coords(lhs, rhs)


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    return _ext().lookup_coords(coords, queries)


def build_kernel_map(
    coords: mx.array,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    return _ext().build_kernel_map(
        coords, *kernel_size, *stride, *padding, *dilation
    )


def build_generative_map(
    coords: mx.array,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    return _ext().build_generative_map(coords, *kernel_size, *stride)


def build_transposed_kernel_map(
    coords: mx.array,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    return _ext().build_transposed_kernel_map(
        coords, *kernel_size, *stride, *padding, *dilation
    )


def conv3d_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().conv3d_feats(
            feats,
            weight,
            maps,
            kernels,
            out_rows,
        )
    return _ext().conv3d_feats(
        feats,
        weight,
        maps,
        kernels,
        out_rows,
        stream=stream,
    )


def conv3d_subm_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    center_kernel: int,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().conv3d_subm_feats(
            feats,
            weight,
            maps,
            kernels,
            center_kernel,
        )
    return _ext().conv3d_subm_feats(
        feats,
        weight,
        maps,
        kernels,
        center_kernel,
        stream=stream,
    )


def conv3d_residual_feats(
    base: mx.array,
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    offsets: mx.array,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().conv3d_residual_feats(
            base,
            feats,
            weight,
            maps,
            kernels,
            offsets,
        )
    return _ext().conv3d_residual_feats(
        base,
        feats,
        weight,
        maps,
        kernels,
        offsets,
        stream=stream,
    )


def pool3d_feats(
    feats: mx.array,
    maps: mx.array,
    kernels: mx.array,
    offsets: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().pool3d_feats(
            feats,
            maps,
            kernels,
            offsets,
            out_rows,
        )
    return _ext().pool3d_feats(
        feats,
        maps,
        kernels,
        offsets,
        out_rows,
        stream=stream,
    )


def max_pool3d_feats(
    feats: mx.array,
    maps: mx.array,
    kernels: mx.array,
    offsets: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array:
    if stream is None:
        return _ext().max_pool3d_feats(
            feats,
            maps,
            kernels,
            offsets,
            out_rows,
        )
    return _ext().max_pool3d_feats(
        feats,
        maps,
        kernels,
        offsets,
        out_rows,
        stream=stream,
    )
