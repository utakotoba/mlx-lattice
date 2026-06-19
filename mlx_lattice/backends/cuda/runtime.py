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
        'build_generative_relation',
        'build_kernel_relation',
        'build_knn_relation',
        'build_radius_relation',
        'build_target_kernel_relation',
        'build_transposed_kernel_relation',
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
type Triple = tuple[int, int, int]


def runtime_available() -> bool:
    return not _availability_issues()


def selected() -> bool:
    return runtime_available() and mx.default_device() == mx.gpu


def info() -> dict[str, Any]:
    issues = _availability_issues()
    return {
        'available': not issues,
        'api': 'mx.fast.precompiled_cuda_kernel',
        'artifact_package': _ARTIFACT_PACKAGE,
        'artifacts': _ARTIFACTS,
        'missing': issues,
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
    return _metadata_tuple((out[0], out[1]))


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
    return _metadata_tuple((out[0], out[1]))


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
    return _metadata_tuple((out[0], out[1]))


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    _require_int32_coords(coords, 'coords')
    _require_int32_coords(queries, 'queries')
    return _metadata(
        _run(
            artifact='coords.ptx',
            name='lookup_coords_i32',
            inputs=[coords, queries],
            output_shapes=[(queries.shape[0],)],
            output_dtypes=[mx.int32],
            scalars=[coords.shape[0], queries.shape[0]],
            grid=_grid_1d(queries.shape[0]),
            threadgroup=(256, 1, 1),
            init_value=-1.0,
        )[0]
    )


def morton_codes(coords: mx.array) -> mx.array:
    _require_int32_coords(coords, 'coords')
    return _metadata(
        _run(
            artifact='coords.ptx',
            name='morton_codes_i32',
            inputs=[coords],
            output_shapes=[(coords.shape[0],)],
            output_dtypes=[mx.int64],
            scalars=[coords.shape[0]],
            grid=_grid_1d(coords.shape[0]),
            threadgroup=(256, 1, 1),
        )[0]
    )


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
    return _metadata_tuple((out[0], out[1], out[2]))


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
    return _metadata_tuple((out[0], out[1], out[2], out[3]))


def child_coords_from_indices(
    parent_coords: mx.array,
    child_indices: mx.array,
) -> mx.array:
    _require_int32_coords(parent_coords, 'parent_coords')
    return _metadata(
        _run(
            artifact='coords.ptx',
            name='child_coords_from_indices_i32',
            inputs=[parent_coords, child_indices],
            output_shapes=[parent_coords.shape],
            output_dtypes=[mx.int32],
            scalars=[parent_coords.shape[0]],
            grid=_grid_1d(parent_coords.shape[0]),
            threadgroup=(256, 1, 1),
        )[0]
    )


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
    return _metadata_tuple((out[0], out[1], out[2], out[3]))


# MARK: - relations


def build_kernel_relation(
    coords: mx.array,
    active_rows: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    return _generic_kernel_relation(
        coords=coords,
        active_rows=active_rows,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        op=0,
        direct=False,
        out_capacity=coords.shape[0],
        edge_capacity=coords.shape[0] * _kernel_count(kernel_size),
    )


def build_transposed_kernel_relation(
    coords: mx.array,
    active_rows: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    offsets = _kernel_offsets(kernel_size, dilation)
    return _generic_kernel_relation(
        coords=coords,
        active_rows=active_rows,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        op=1,
        direct=_can_use_direct_transposed_relation(offsets, stride),
        out_capacity=coords.shape[0] * len(offsets),
        edge_capacity=coords.shape[0] * len(offsets),
    )


def build_generative_relation(
    coords: mx.array,
    active_rows: mx.array,
    kernel_size: Triple,
    stride: Triple,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    _require_int32_coords(coords, 'coords')
    offsets = _kernel_offsets(kernel_size, (1, 1, 1))
    edge_capacity = coords.shape[0] * len(offsets)
    out = _run(
        artifact='coords.ptx',
        name='generative_kernel_relation_full_i32',
        inputs=[coords, _offset_array(offsets), active_rows],
        output_shapes=[
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity + 1,),
            (edge_capacity, 4),
            (2,),
            (coords.shape[0] + 1,),
            (edge_capacity,),
            (len(offsets) + 1,),
            (edge_capacity,),
        ],
        output_dtypes=[
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
        ],
        scalars=[coords.shape[0], len(offsets), *stride],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return _relation_tuple(out)


def build_target_kernel_relation(
    coords: mx.array,
    active_rows: mx.array,
    target_coords: mx.array,
    target_active_rows: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    _require_int32_coords(coords, 'coords')
    _require_int32_coords(target_coords, 'target_coords')
    offsets = _kernel_offsets(kernel_size, dilation)
    edge_capacity = target_coords.shape[0] * len(offsets)
    out = _run(
        artifact='coords.ptx',
        name='target_kernel_relation_full_i32',
        inputs=[
            coords,
            _offset_array(offsets),
            active_rows,
            target_coords,
            target_active_rows,
        ],
        output_shapes=[
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity,),
            (target_coords.shape[0] + 1,),
            (2,),
            (coords.shape[0] + 1,),
            (edge_capacity,),
            (len(offsets) + 1,),
            (edge_capacity,),
        ],
        output_dtypes=[
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
        ],
        scalars=[
            coords.shape[0],
            target_coords.shape[0],
            len(offsets),
            *stride,
            *padding,
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return _metadata_tuple(
        (
            out[0],
            out[1],
            out[2],
            out[3],
            target_coords,
            out[4],
            out[5],
            out[6],
            out[7],
            out[8],
        )
    )


def build_knn_relation(
    source_coords: mx.array,
    source_active_rows: mx.array,
    query_coords: mx.array,
    query_active_rows: mx.array,
    k: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    return _neighbor_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        max_neighbors=int(k),
        radius_squared=0.0,
        op=0,
    )


def build_radius_relation(
    source_coords: mx.array,
    source_active_rows: mx.array,
    query_coords: mx.array,
    query_active_rows: mx.array,
    radius: float,
    max_neighbors: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    return _neighbor_relation(
        source_coords,
        source_active_rows,
        query_coords,
        query_active_rows,
        max_neighbors=_radius_neighbor_capacity(radius)
        if max_neighbors == 0
        else int(max_neighbors),
        radius_squared=float(radius) * float(radius),
        op=1,
    )


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
    cotangent = _single(cotangents)
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
        mx.stop_gradient(inverse_rows),
        mx.stop_gradient(voxel_counts),
        mx.stop_gradient(active_rows),
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
    cotangent = _single(cotangents)
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
            mx.stop_gradient(weights),
            mx.stop_gradient(in_rows),
            mx.stop_gradient(out_rows),
            mx.stop_gradient(kernel_ids),
            mx.stop_gradient(counts),
            mx.stop_gradient(row_offsets),
            out_capacity,
            n_kernels,
        )
    if weight_tangent is not None:
        component = _sparse_conv_forward_run(
            mx.stop_gradient(feats),
            weight_tangent,
            mx.stop_gradient(in_rows),
            mx.stop_gradient(out_rows),
            mx.stop_gradient(kernel_ids),
            mx.stop_gradient(counts),
            mx.stop_gradient(row_offsets),
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
    cotangent = _single(cotangents)
    pooled = _single(outputs)
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
        mx.stop_gradient(feats),
        mx.stop_gradient(in_rows),
        mx.stop_gradient(out_rows),
        mx.stop_gradient(kernel_ids),
        mx.stop_gradient(row_offsets),
        mx.stop_gradient(counts),
        reduce_id,
        out_capacity,
    )
    return _run(
        artifact='pool.ptx',
        name='sparse_pool_relation_jvp_f32_i32',
        inputs=[
            tangent,
            mx.stop_gradient(feats),
            mx.stop_gradient(pooled),
            mx.stop_gradient(in_rows),
            mx.stop_gradient(out_rows),
            mx.stop_gradient(kernel_ids),
            mx.stop_gradient(row_offsets),
            mx.stop_gradient(counts),
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
        grid=(out_capacity * 128, feats.shape[1], 1),
        threadgroup=(128, 1, 1),
    )[0]


# MARK: - relation helpers


def _generic_kernel_relation(
    *,
    coords: mx.array,
    active_rows: mx.array,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    op: int,
    direct: bool,
    out_capacity: int,
    edge_capacity: int,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    _require_int32_coords(coords, 'coords')
    offsets = _kernel_offsets(kernel_size, dilation)
    out = _run(
        artifact='coords.ptx',
        name='generic_kernel_relation_full_i32',
        inputs=[coords, _offset_array(offsets), active_rows],
        output_shapes=[
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity,),
            (out_capacity + 1,),
            (out_capacity, 4),
            (2,),
            (coords.shape[0] + 1,),
            (edge_capacity,),
            (len(offsets) + 1,),
            (edge_capacity,),
        ],
        output_dtypes=[
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
            mx.int32,
        ],
        scalars=[
            op,
            coords.shape[0],
            len(offsets),
            *stride,
            *padding,
            bool(direct),
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return _relation_tuple(out)


def _neighbor_relation(
    source_coords: mx.array,
    source_active_rows: mx.array,
    query_coords: mx.array,
    query_active_rows: mx.array,
    *,
    max_neighbors: int,
    radius_squared: float,
    op: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    _require_int32_coords(source_coords, 'source_coords')
    _require_int32_coords(query_coords, 'query_coords')
    edge_capacity = query_coords.shape[0] * max_neighbors
    out = _run(
        artifact='coords.ptx',
        name='neighbor_relation_i32',
        inputs=[
            source_coords,
            query_coords,
            source_active_rows,
            query_active_rows,
        ],
        output_shapes=[
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity,),
            (edge_capacity,),
            (query_coords.shape[0] + 1,),
            (2,),
        ],
        output_dtypes=[
            mx.int32,
            mx.int32,
            mx.int32,
            mx.float32,
            mx.int32,
            mx.int32,
        ],
        scalars=[
            op,
            source_coords.shape[0],
            query_coords.shape[0],
            max_neighbors,
            radius_squared,
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return _metadata_tuple((out[0], out[1], out[2], out[3], out[4], out[5]))


def _relation_tuple(
    out: list[mx.array],
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]:
    return _metadata_tuple(
        (
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5],
            out[6],
            out[7],
            out[8],
            out[9],
        )
    )


def _kernel_offsets(
    kernel_size: Triple,
    dilation: Triple,
) -> tuple[Triple, ...]:
    axes = []
    for size in kernel_size:
        if size % 2 == 1:
            radius = size // 2
            axes.append(range(-radius, radius + 1))
        else:
            axes.append(range(size))
    return tuple(
        (int(x * dilation[0]), int(y * dilation[1]), int(z * dilation[2]))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def _kernel_count(kernel_size: Triple) -> int:
    return int(kernel_size[0] * kernel_size[1] * kernel_size[2])


def _offset_array(offsets: tuple[Triple, ...]) -> mx.array:
    return mx.array(offsets, dtype=mx.int32)


def _can_use_direct_transposed_relation(
    offsets: tuple[Triple, ...],
    stride: Triple,
) -> bool:
    if not offsets:
        return False
    for axis in range(3):
        values = [offset[axis] for offset in offsets]
        if stride[axis] <= max(values) - min(values):
            return False
    return True


def _radius_neighbor_capacity(radius: float) -> int:
    import math

    limit = math.ceil(radius)
    radius_squared = radius * radius
    count = 0
    for dz in range(-limit, limit + 1):
        for dy in range(-limit, limit + 1):
            for dx in range(-limit, limit + 1):
                if dx * dx + dy * dy + dz * dz <= radius_squared:
                    count += 1
    return max(count, 1)


# MARK: - low-level helpers


def _single(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return value[0]
    return value


def _metadata(value: mx.array) -> mx.array:
    return mx.stop_gradient(value)


def _metadata_tuple[T: tuple[Any, ...] | list[Any]](values: T) -> T:
    return type(values)(
        _metadata(value) if isinstance(value, mx.array) else value
        for value in values
    )


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
    total = max(int(elements), 1)
    return (((total + block - 1) // block) * block, 1, 1)


def _require_int32_coords(value: mx.array, name: str) -> None:
    if value.dtype != mx.int32 or value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(
            f'CUDA coordinate kernels require {name} to have shape '
            '(N, 4) and int32 dtype.'
        )


def _availability_issues() -> tuple[str, ...]:
    issues = []
    cuda = getattr(mx, 'cuda', None)
    if cuda is None:
        issues.append('mlx.core.cuda is unavailable')
    elif not cuda.is_available():
        issues.append('mlx.core.cuda reports unavailable')
    if not hasattr(mx.fast, 'precompiled_cuda_kernel'):
        issues.append('mx.fast.precompiled_cuda_kernel is unavailable')
    for artifact in _ARTIFACTS:
        if not _artifact_exists(artifact):
            issues.append(f'missing artifact {artifact}')
    return tuple(issues)


def _artifact_exists(name: str) -> bool:
    try:
        return resources.files(_ARTIFACT_PACKAGE).joinpath(name).is_file()
    except ModuleNotFoundError:
        return False


@cache
def _artifact_bytes(name: str) -> bytes:
    return resources.files(_ARTIFACT_PACKAGE).joinpath(name).read_bytes()
