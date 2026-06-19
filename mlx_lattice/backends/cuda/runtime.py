from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from importlib import resources
from typing import Any, Literal, cast

import mlx.core as mx

_ARTIFACT_PACKAGE = 'mlx_lattice.backends.cuda.artifacts'
_ARTIFACTS = ('coords.ptx', 'conv.ptx', 'pool.ptx')
_IMPLEMENTED_OPS = frozenset(
    {
        'child_coords_from_indices',
        'downsample_coords',
        'intersection_coords',
        'lookup_coords',
        'morton_codes',
        'occupancy_downsample',
        'occupancy_expand',
        'sparse_conv_features',
        'sparse_pool_features',
        'sparse_quantize',
        'union_coords',
        'voxelize_features',
    }
)

type Reduction = Literal['sum', 'max', 'avg']
type VoxelReduction = Literal['sum', 'mean']


def runtime_available() -> bool:
    cuda = getattr(mx, 'cuda', None)
    return bool(
        cuda is not None
        and cuda.is_available()
        and hasattr(mx.fast, 'precompiled_cuda_kernel')
        and all(_artifact_exists(name) for name in _ARTIFACTS)
    )


def selected() -> bool:
    return runtime_available() and mx.default_device() == mx.gpu


def info() -> dict[str, Any]:
    return {
        'available': runtime_available(),
        'api': 'mx.fast.precompiled_cuda_kernel',
        'artifact_package': _ARTIFACT_PACKAGE,
        'artifacts': _ARTIFACTS,
        'implemented_ops': tuple(sorted(_IMPLEMENTED_OPS)),
    }


# MARK: - coordinate primitives


def downsample_coords(
    coords: mx.array,
    stride: tuple[int, int, int],
) -> tuple[mx.array, mx.array]:
    _require_int32_coords(coords, 'coords')
    out = _run(
        artifact='coords.ptx',
        name='set_coords_i32',
        inputs=[coords, coords],
        output_shapes=[coords.shape, (1,)],
        output_dtypes=[mx.int32, mx.int32],
        scalars=[0, *stride, coords.shape[0], coords.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return out[0], out[1]


def union_coords(lhs: mx.array, rhs: mx.array) -> tuple[mx.array, mx.array]:
    _require_int32_coords(lhs, 'lhs')
    _require_int32_coords(rhs, 'rhs')
    capacity = lhs.shape[0] + rhs.shape[0]
    out = _run(
        artifact='coords.ptx',
        name='set_coords_i32',
        inputs=[lhs, rhs],
        output_shapes=[(capacity, 4), (1,)],
        output_dtypes=[mx.int32, mx.int32],
        scalars=[1, 1, 1, 1, lhs.shape[0], rhs.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return out[0], out[1]


def intersection_coords(
    lhs: mx.array,
    rhs: mx.array,
) -> tuple[mx.array, mx.array]:
    _require_int32_coords(lhs, 'lhs')
    _require_int32_coords(rhs, 'rhs')
    out = _run(
        artifact='coords.ptx',
        name='set_coords_i32',
        inputs=[lhs, rhs],
        output_shapes=[(lhs.shape[0], 4), (1,)],
        output_dtypes=[mx.int32, mx.int32],
        scalars=[2, 1, 1, 1, lhs.shape[0], rhs.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return out[0], out[1]


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    _require_int32_coords(coords, 'coords')
    _require_int32_coords(queries, 'queries')
    return _run(
        artifact='coords.ptx',
        name='lookup_coords_i32',
        inputs=[coords, queries],
        output_shapes=[(queries.shape[0],)],
        output_dtypes=[mx.int32],
        scalars=[coords.shape[0], queries.shape[0]],
        grid=_grid_1d(queries.shape[0]),
        threadgroup=(256, 1, 1),
    )[0]


def morton_codes(coords: mx.array) -> mx.array:
    _require_int32_coords(coords, 'coords')
    return _run(
        artifact='coords.ptx',
        name='morton_codes_i32',
        inputs=[coords],
        output_shapes=[(coords.shape[0],)],
        output_dtypes=[mx.int64],
        scalars=[coords.shape[0]],
        grid=_grid_1d(coords.shape[0]),
        threadgroup=(256, 1, 1),
    )[0]


def occupancy_downsample(
    coords: mx.array,
    active_rows: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    _require_int32_coords(coords, 'coords')
    out = _run(
        artifact='coords.ptx',
        name='occupancy_downsample_i32',
        inputs=[coords, active_rows],
        output_shapes=[coords.shape, (1,), (coords.shape[0],)],
        output_dtypes=[mx.int32, mx.int32, mx.int32],
        scalars=[coords.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return out[0], out[1], out[2]


def occupancy_expand(
    coords: mx.array,
    active_rows: mx.array,
    occupancy: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    _require_int32_coords(coords, 'coords')
    capacity = coords.shape[0] * 8
    out = _run(
        artifact='coords.ptx',
        name='occupancy_expand_i32',
        inputs=[coords, active_rows, occupancy],
        output_shapes=[(capacity, 4), (1,), (capacity,), (capacity,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.int32],
        scalars=[coords.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return out[0], out[1], out[2], out[3]


def child_coords_from_indices(
    parent_coords: mx.array,
    child_indices: mx.array,
) -> mx.array:
    _require_int32_coords(parent_coords, 'parent_coords')
    return _run(
        artifact='coords.ptx',
        name='child_coords_from_indices_i32',
        inputs=[parent_coords, child_indices],
        output_shapes=[parent_coords.shape],
        output_dtypes=[mx.int32],
        scalars=[parent_coords.shape[0]],
        grid=_grid_1d(parent_coords.shape[0]),
        threadgroup=(256, 1, 1),
    )[0]


# MARK: - quantization


def sparse_quantize(
    points: mx.array,
    batch_indices: mx.array,
    active_rows: mx.array,
    voxel_size: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if (
        points.dtype != mx.float32
        or points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError('CUDA sparse_quantize requires float32 points.')
    out = _run(
        artifact='coords.ptx',
        name='sparse_quantize_i32',
        inputs=[points, batch_indices, active_rows],
        output_shapes=[
            (points.shape[0], 4),
            (1,),
            (points.shape[0],),
            (points.shape[0],),
        ],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.int32],
        scalars=[*voxel_size, *origin, points.shape[0]],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return out[0], out[1], out[2], out[3]


@mx.custom_function
def _voxelize_features(
    feats: mx.array,
    inverse_rows: mx.array,
    voxel_counts: mx.array,
    active_rows: mx.array,
    reduce_id: int,
    voxel_rows: int,
) -> mx.array:
    return _voxelize_features_run(
        feats,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduce_id,
        voxel_rows,
    )


@_voxelize_features.vjp
def _voxelize_features_vjp(primals, cotangents, outputs):
    (
        feats,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduce_id,
        voxel_rows,
    ) = primals
    (cotangent,) = cotangents
    del outputs, voxel_rows
    grad_feats = _run(
        artifact='coords.ptx',
        name='voxelize_feature_grad_f32',
        inputs=[cotangent, inverse_rows, voxel_counts, active_rows],
        output_shapes=[feats.shape],
        output_dtypes=[feats.dtype],
        scalars=[
            reduce_id,
            feats.shape[0],
            voxel_counts.shape[0],
            feats.shape[1],
        ],
        grid=_grid_1d(feats.size),
        threadgroup=(256, 1, 1),
    )[0]
    return grad_feats, None, None, None, None, None


@_voxelize_features.jvp
def _voxelize_features_jvp(primals, tangents):
    _, inverse_rows, voxel_counts, active_rows, reduce_id, voxel_rows = (
        primals
    )
    tangent, *_ = tangents
    if tangent is None:
        return mx.zeros((voxel_rows, 0), dtype=mx.float32)
    return _voxelize_features_run(
        tangent,
        inverse_rows,
        voxel_counts,
        active_rows,
        reduce_id,
        voxel_rows,
    )


def voxelize_features(
    feats: mx.array,
    inverse_rows: mx.array,
    voxel_counts: mx.array,
    active_rows: mx.array,
    reduction: VoxelReduction,
) -> mx.array:
    reduce_id = 1 if reduction == 'mean' else 0
    return cast(
        mx.array,
        _voxelize_features(
            feats,
            inverse_rows,
            voxel_counts,
            active_rows,
            reduce_id,
            voxel_counts.shape[0],
        ),
    )


def _voxelize_features_run(
    feats: mx.array,
    inverse_rows: mx.array,
    voxel_counts: mx.array,
    active_rows: mx.array,
    reduce_id: int,
    voxel_rows: int,
) -> mx.array:
    return _run(
        artifact='coords.ptx',
        name='voxelize_features_f32',
        inputs=[feats, inverse_rows, voxel_counts, active_rows],
        output_shapes=[(voxel_rows, feats.shape[1])],
        output_dtypes=[feats.dtype],
        scalars=[reduce_id, feats.shape[0], voxel_rows, feats.shape[1]],
        grid=_grid_1d(feats.size),
        threadgroup=(256, 1, 1),
        init_value=0.0,
    )[0]


# MARK: - feature execution


@mx.custom_function
def _sparse_conv_features(
    feats: mx.array,
    weights: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    counts: mx.array,
    row_offsets: mx.array,
    in_row_offsets: mx.array,
    in_edge_ids: mx.array,
    kernel_row_offsets: mx.array,
    kernel_edge_ids: mx.array,
    out_capacity: int,
    n_kernels: int,
) -> mx.array:
    return _sparse_conv_forward_run(
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        out_capacity,
        n_kernels,
    )


@_sparse_conv_features.vjp
def _sparse_conv_features_vjp(primals, cotangents, outputs):
    (
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        in_row_offsets,
        in_edge_ids,
        kernel_row_offsets,
        kernel_edge_ids,
        out_capacity,
        n_kernels,
    ) = primals
    (cotangent,) = cotangents
    del outputs
    shape = _conv_shape(feats, weights, out_capacity, n_kernels)
    grad_feats = _run(
        artifact='conv.ptx',
        name=_conv_kernel_name('sparse_conv_input_grad', feats.dtype),
        inputs=[
            cotangent,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            in_row_offsets,
            in_edge_ids,
        ],
        output_shapes=[feats.shape],
        output_dtypes=[feats.dtype],
        scalars=_conv_scalars(
            shape, feats, cotangent, weights, feats.shape
        ),
        grid=_grid_1d(feats.size),
        threadgroup=(256, 1, 1),
    )[0]
    grad_weights = _run(
        artifact='conv.ptx',
        name=_conv_kernel_name('sparse_conv_weight_grad', feats.dtype),
        inputs=[
            feats,
            cotangent,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            kernel_row_offsets,
            kernel_edge_ids,
        ],
        output_shapes=[weights.shape],
        output_dtypes=[weights.dtype],
        scalars=_conv_scalars(
            shape, feats, cotangent, weights, weights.shape
        ),
        grid=_grid_1d(weights.size),
        threadgroup=(256, 1, 1),
    )[0]
    return (
        grad_feats,
        grad_weights,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@_sparse_conv_features.jvp
def _sparse_conv_features_jvp(primals, tangents):
    (
        feats,
        weights,
        in_rows,
        out_rows,
        kernel_ids,
        counts,
        row_offsets,
        *_,
        out_capacity,
        n_kernels,
    ) = primals
    feat_tangent, weight_tangent, *_ = tangents
    out = None
    if feat_tangent is not None:
        out = _sparse_conv_forward_run(
            feat_tangent,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            out_capacity,
            n_kernels,
        )
    if weight_tangent is not None:
        component = _sparse_conv_forward_run(
            feats,
            weight_tangent,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            out_capacity,
            n_kernels,
        )
        out = component if out is None else out + component
    if out is None:
        out_channels = _conv_shape(feats, weights, out_capacity, n_kernels)[
            5
        ]
        return mx.zeros((out_capacity, out_channels), dtype=feats.dtype)
    return out


def sparse_conv_features(
    feats: mx.array,
    weights: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    counts: mx.array,
    row_offsets: mx.array,
    in_row_offsets: mx.array,
    in_edge_ids: mx.array,
    kernel_row_offsets: mx.array,
    kernel_edge_ids: mx.array,
    out_capacity: int,
    n_kernels: int,
) -> mx.array:
    return cast(
        mx.array,
        _sparse_conv_features(
            feats,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
            in_row_offsets,
            in_edge_ids,
            kernel_row_offsets,
            kernel_edge_ids,
            int(out_capacity),
            int(n_kernels),
        ),
    )


def _sparse_conv_forward_run(
    feats: mx.array,
    weights: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    counts: mx.array,
    row_offsets: mx.array,
    out_capacity: int,
    n_kernels: int,
) -> mx.array:
    shape = _conv_shape(feats, weights, out_capacity, n_kernels)
    out_shape = (out_capacity, shape[5])
    return _run(
        artifact='conv.ptx',
        name=_conv_kernel_name('sparse_conv_forward', feats.dtype),
        inputs=[
            feats,
            weights,
            in_rows,
            out_rows,
            kernel_ids,
            counts,
            row_offsets,
        ],
        output_shapes=[out_shape],
        output_dtypes=[feats.dtype],
        scalars=_conv_scalars(shape, feats, None, weights, out_shape),
        grid=_grid_1d(out_capacity * shape[5]),
        threadgroup=(256, 1, 1),
    )[0]


@mx.custom_function
def _sparse_pool_features(
    feats: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    row_offsets: mx.array,
    counts: mx.array,
    in_row_offsets: mx.array,
    in_edge_ids: mx.array,
    input_exclusive: bool,
    reduce_id: int,
    out_capacity: int,
    n_kernels: int,
) -> mx.array:
    del in_row_offsets, in_edge_ids, input_exclusive, n_kernels
    return _sparse_pool_forward_run(
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        reduce_id,
        out_capacity,
    )


@_sparse_pool_features.vjp
def _sparse_pool_features_vjp(primals, cotangents, outputs):
    (
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
        input_exclusive,
        reduce_id,
        out_capacity,
        n_kernels,
    ) = primals
    (cotangent,) = cotangents
    (pooled,) = outputs
    if input_exclusive:
        kernel_name = 'sparse_pool_relation_exclusive_input_grad_f32_i32'
    else:
        kernel_name = (
            'sparse_pool_relation_max_input_grad_f32_i32'
            if reduce_id == 1
            else 'sparse_pool_relation_sum_avg_input_grad_f32_i32'
        )
    grad_feats = _run(
        artifact='pool.ptx',
        name=kernel_name,
        inputs=[
            cotangent,
            feats,
            pooled,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
            in_row_offsets,
            in_edge_ids,
        ],
        output_shapes=[feats.shape],
        output_dtypes=[feats.dtype],
        scalars=[
            reduce_id,
            feats.shape[0],
            out_capacity,
            n_kernels,
            feats.shape[1],
            cotangent.shape[1],
            1,
            feats.shape[1],
            1,
            pooled.shape[1],
            1,
        ],
        grid=_grid_1d(feats.size),
        threadgroup=(256, 1, 1),
    )[0]
    return (
        grad_feats,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@_sparse_pool_features.jvp
def _sparse_pool_features_jvp(primals, tangents):
    (
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        in_row_offsets,
        in_edge_ids,
        input_exclusive,
        reduce_id,
        out_capacity,
        n_kernels,
    ) = primals
    tangent, *_ = tangents
    del in_row_offsets, in_edge_ids, input_exclusive
    if tangent is None:
        return mx.zeros((out_capacity, feats.shape[1]), dtype=feats.dtype)
    pooled = _sparse_pool_forward_run(
        feats,
        in_rows,
        out_rows,
        kernel_ids,
        row_offsets,
        counts,
        reduce_id,
        out_capacity,
    )
    return _run(
        artifact='pool.ptx',
        name='sparse_pool_relation_jvp_f32_i32',
        inputs=[
            tangent,
            feats,
            pooled,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
        ],
        output_shapes=[pooled.shape],
        output_dtypes=[feats.dtype],
        scalars=[
            reduce_id,
            feats.shape[0],
            out_capacity,
            n_kernels,
            feats.shape[1],
            tangent.shape[1],
            1,
            feats.shape[1],
            1,
            pooled.shape[1],
            1,
        ],
        grid=_grid_1d(pooled.size),
        threadgroup=(256, 1, 1),
    )[0]


def sparse_pool_features(
    feats: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    row_offsets: mx.array,
    counts: mx.array,
    in_row_offsets: mx.array,
    in_edge_ids: mx.array,
    input_exclusive: bool,
    reduce: Reduction,
    out_capacity: int,
    n_kernels: int,
) -> mx.array:
    return cast(
        mx.array,
        _sparse_pool_features(
            feats,
            in_rows,
            out_rows,
            kernel_ids,
            row_offsets,
            counts,
            in_row_offsets,
            in_edge_ids,
            bool(input_exclusive),
            _pool_reduce_id(reduce),
            int(out_capacity),
            int(n_kernels),
        ),
    )


def _sparse_pool_forward_run(
    feats: mx.array,
    in_rows: mx.array,
    out_rows: mx.array,
    kernel_ids: mx.array,
    row_offsets: mx.array,
    counts: mx.array,
    reduce_id: int,
    out_capacity: int,
) -> mx.array:
    del out_rows, kernel_ids
    name = {
        0: 'sparse_pool_relation_block_sum_f32_i32',
        1: 'sparse_pool_relation_block_max_f32_i32',
        2: 'sparse_pool_relation_block_avg_f32_i32',
    }[reduce_id]
    return _run(
        artifact='pool.ptx',
        name=name,
        inputs=[feats, in_rows, row_offsets, counts],
        output_shapes=[(out_capacity, feats.shape[1])],
        output_dtypes=[feats.dtype],
        scalars=[out_capacity, feats.shape[1], feats.shape[1], 1],
        grid=(out_capacity, feats.shape[1], 1),
        threadgroup=(128, 1, 1),
    )[0]


# MARK: - low-level helpers


def _run(
    *,
    artifact: str,
    name: str,
    inputs: list[mx.array],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[mx.Dtype],
    scalars: Sequence[bool | int | float],
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    init_value: float | None = None,
) -> list[mx.array]:
    return mx.fast.precompiled_cuda_kernel(
        name=name,
        compiled_source=_artifact_bytes(artifact),
        inputs=inputs,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        scalars=list(scalars),
        grid=grid,
        threadgroup=threadgroup,
        init_value=init_value,
        ensure_row_contiguous=True,
    )


def _conv_shape(
    feats: mx.array,
    weights: mx.array,
    out_capacity: int,
    n_kernels: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    mapped = weights.ndim == 3
    return (
        n_kernels * feats.shape[0],
        feats.shape[0],
        out_capacity,
        n_kernels,
        feats.shape[1],
        weights.shape[2] if mapped else weights.shape[0],
        0 if mapped else 1,
        n_kernels if mapped else weights.shape[1],
        1 if mapped else weights.shape[2],
        1 if mapped else weights.shape[3],
    )


def _conv_scalars(
    shape: tuple[int, int, int, int, int, int, int, int, int, int],
    feats: mx.array,
    cotangent: mx.array | None,
    weights: mx.array,
    out_shape: tuple[int, ...],
) -> list[bool | int | float]:
    cot_s0 = shape[5] if cotangent is None else cotangent.shape[1]
    cot_s1 = 1
    weight_strides = _weight_strides(weights)
    out_strides = _dense_strides(out_shape)
    return [
        *shape,
        feats.shape[1],
        1,
        cot_s0,
        cot_s1,
        *weight_strides,
        *out_strides,
    ]


def _weight_strides(weights: mx.array) -> tuple[int, int, int, int, int]:
    if weights.ndim == 3:
        return (
            weights.shape[1] * weights.shape[2],
            weights.shape[2],
            1,
            1,
            1,
        )
    return (
        weights.shape[1]
        * weights.shape[2]
        * weights.shape[3]
        * weights.shape[4],
        weights.shape[2] * weights.shape[3] * weights.shape[4],
        weights.shape[3] * weights.shape[4],
        weights.shape[4],
        1,
    )


def _dense_strides(
    shape: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    strides: list[int] = []
    running = 1
    for size in reversed(shape):
        strides.append(running)
        running *= size
    padded = [*reversed(strides), 1, 1, 1, 1, 1]
    return (padded[0], padded[1], padded[2], padded[3], padded[4])


def _conv_kernel_name(prefix: str, dtype: mx.Dtype) -> str:
    if dtype == mx.float32:
        return f'{prefix}_f32_flat'
    if dtype == mx.float16:
        return f'{prefix}_f16_flat'
    raise ValueError(
        'CUDA sparse convolution supports float32 and float16.'
    )


def _pool_reduce_id(reduce: Reduction) -> int:
    if reduce == 'sum':
        return 0
    if reduce == 'max':
        return 1
    if reduce == 'avg':
        return 2
    raise ValueError("reduce must be 'sum', 'max', or 'avg'.")


def _grid_1d(elements: int, block: int = 256) -> tuple[int, int, int]:
    return (max((int(elements) + block - 1) // block, 1), 1, 1)


def _require_int32_coords(value: mx.array, name: str) -> None:
    if value.dtype != mx.int32 or value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(
            f'CUDA {name} must have shape (N, 4) and int32 dtype.'
        )


def _artifact_exists(name: str) -> bool:
    try:
        return resources.files(_ARTIFACT_PACKAGE).joinpath(name).is_file()
    except ModuleNotFoundError:
        return False


@cache
def _artifact_bytes(name: str) -> bytes:
    return resources.files(_ARTIFACT_PACKAGE).joinpath(name).read_bytes()
