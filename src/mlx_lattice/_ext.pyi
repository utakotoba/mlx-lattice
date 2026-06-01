from typing import Any

import mlx.core as mx

def version() -> str: ...
def capabilities() -> dict[str, bool]: ...
def downsample_coords(
    coords: mx.array,
    sx: int,
    sy: int,
    sz: int,
) -> mx.array: ...
def build_kernel_map(
    coords: mx.array,
    kx: int,
    ky: int,
    kz: int,
    sx: int,
    sy: int,
    sz: int,
    px: int,
    py: int,
    pz: int,
    dx: int,
    dy: int,
    dz: int,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]: ...
def build_generative_map(
    coords: mx.array,
    kx: int,
    ky: int,
    kz: int,
    sx: int,
    sy: int,
    sz: int,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]: ...
def build_transposed_kernel_map(
    coords: mx.array,
    kx: int,
    ky: int,
    kz: int,
    sx: int,
    sy: int,
    sz: int,
    px: int,
    py: int,
    pz: int,
    dx: int,
    dy: int,
    dz: int,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]: ...
def conv3d_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array: ...
def conv3d_subm_feats(
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    center_kernel: int,
    *,
    stream: Any | None = None,
) -> mx.array: ...
def conv3d_residual_feats(
    base: mx.array,
    feats: mx.array,
    weight: mx.array,
    maps: mx.array,
    kernels: mx.array,
    offsets: mx.array,
    *,
    stream: Any | None = None,
) -> mx.array: ...
def pool3d_feats(
    feats: mx.array,
    maps: mx.array,
    kernels: mx.array,
    offsets: mx.array,
    out_rows: int,
    *,
    stream: Any | None = None,
) -> mx.array: ...
