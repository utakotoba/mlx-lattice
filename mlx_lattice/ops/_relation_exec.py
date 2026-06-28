from __future__ import annotations

import weakref

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.quantized import QuantizedWeight
from mlx_lattice.core.relations import KernelRelation

_PACKED_WEIGHT_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[mx.array], tuple[int, ...], mx.Dtype, mx.array
    ],
] = {}


def sparse_quantized_conv_features_from_relation(
    feats: mx.array,
    weight: QuantizedWeight,
    relation: KernelRelation,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    if relation.n_kernels != weight.weight.shape[0]:
        raise ValueError(
            'quantized weight kernel rows must match the relation.'
        )
    return ext.sparse_quantized_conv_features(
        feats,
        weight.weight,
        weight.scales,
        weight.biases,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.counts,
        relation.output_csr.row_offsets,
        relation.n_out_capacity,
        relation.n_kernels,
        weight.in_channels,
        weight.out_channels,
        weight.storage_in_channels,
        weight.group_size,
        weight.bits,
    )


def sparse_conv_features_from_relation(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    if _can_use_sorted_implicit_gemm(feats, weight, relation):
        return sparse_conv_features_sorted_from_relation(
            feats, weight, relation
        )
    input_csr = relation.input_csr
    kernel_csr = relation.kernel_csr
    if input_csr.edge_ids is None or kernel_csr.edge_ids is None:
        raise ValueError('kernel relation is missing grouped CSR views.')
    return ext.sparse_conv_features(
        feats,
        weight,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.counts,
        relation.output_csr.row_offsets,
        input_csr.row_offsets,
        input_csr.edge_ids,
        kernel_csr.row_offsets,
        kernel_csr.edge_ids,
        relation.n_out_capacity,
        relation.n_kernels,
    )


def sparse_conv_features_sorted_from_relation(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
    *,
    store_sorted: bool = False,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    if not _can_use_sorted_implicit_gemm(feats, weight, relation):
        raise ValueError(
            'sorted implicit GEMM is not supported for this relation, '
            'feature tensor, or weight tensor.'
        )
    view = relation.require_sorted_implicit_gemm()
    return ext.sparse_conv_features_sorted_implicit_gemm(
        feats,
        _mapped_weight(weight),
        view.sorted_out_in_map,
        view.sorted_kv_out_in_map,
        view.reorder_rows,
        view.tile_masks,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.counts,
        relation.output_csr.row_offsets,
        relation.input_csr.row_offsets,
        relation.in_edge_ids,
        relation.kernel_csr.row_offsets,
        relation.kernel_edge_ids,
        relation.n_out_capacity,
        relation.n_kernels,
        store_sorted=store_sorted,
    )


def sparse_conv_features_sorted_direct_reference_from_relation(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
    *,
    store_sorted: bool = False,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    if not _can_use_sorted_implicit_gemm(feats, weight, relation):
        raise ValueError(
            'sorted direct convolution reference is not supported for this '
            'relation, feature tensor, or weight tensor.'
        )
    view = relation.require_sorted_implicit_gemm()
    return ext.sparse_conv_features_sorted_direct_reference(
        feats,
        _mapped_weight(weight),
        view.sorted_out_in_map,
        view.reorder_rows,
        view.tile_masks,
        relation.n_out_capacity,
        relation.n_kernels,
        store_sorted=store_sorted,
    )


def _can_use_sorted_implicit_gemm(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
) -> bool:
    if relation.contract.kind not in ('forward', 'target'):
        return False
    if feats.dtype != mx.float16 or weight.dtype != mx.float16:
        return False
    if relation.n_kernels != 27:
        return False
    if feats.ndim != 2 or int(feats.shape[1]) not in (32, 64):
        return False
    channels = int(feats.shape[1])
    if weight.ndim == 3:
        return int(weight.shape[0]) == 27 and tuple(weight.shape[1:]) == (
            channels,
            channels,
        )
    return weight.ndim == 5 and tuple(weight.shape) == (
        channels,
        3,
        3,
        3,
        channels,
    )


def _mapped_weight(weight: mx.array) -> mx.array:
    if weight.ndim == 3:
        return weight
    cache_key = id(weight)
    shape = tuple(int(dim) for dim in weight.shape)
    cached = _PACKED_WEIGHT_CACHE.get(cache_key)
    if cached is not None:
        cached_ref, cached_shape, cached_dtype, cached_weight = cached
        if (
            cached_ref() is weight
            and cached_shape == shape
            and cached_dtype == weight.dtype
        ):
            return cached_weight
    channels = int(weight.shape[0])
    packed = mx.contiguous(
        weight.transpose(1, 2, 3, 4, 0).reshape((-1, channels, channels))
    )

    def clear_cached_weight(
        ref: weakref.ReferenceType[mx.array], key: int = cache_key
    ) -> None:
        cached_entry = _PACKED_WEIGHT_CACHE.get(key)
        if cached_entry is not None and cached_entry[0] is ref:
            _PACKED_WEIGHT_CACHE.pop(key, None)

    weight_ref = weakref.ref(weight, clear_cached_weight)
    _PACKED_WEIGHT_CACHE[cache_key] = (
        weight_ref,
        shape,
        weight.dtype,
        packed,
    )
    return packed


def sparse_pool_features_from_relation(
    feats: mx.array,
    relation: KernelRelation,
    *,
    input_exclusive: bool,
    mode: str,
) -> mx.array:
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )
    return ext.sparse_pool_features(
        feats,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.output_csr.row_offsets,
        relation.counts,
        input_exclusive,
        mode,
        relation.n_out_capacity,
        relation.n_kernels,
    )
