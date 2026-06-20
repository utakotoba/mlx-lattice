from __future__ import annotations

import os

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.relations import KernelRelation


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
        view = relation.require_sorted_implicit_gemm()
        return ext.sparse_conv_features_sorted_implicit_gemm(
            feats,
            _mapped_weight(weight),
            view.sorted_out_in_map,
            view.reorder_rows,
            view.tile_masks,
            relation.n_out_capacity,
            relation.n_kernels,
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


def _can_use_sorted_implicit_gemm(
    feats: mx.array,
    weight: mx.array,
    relation: KernelRelation,
) -> bool:
    if os.environ.get('MLX_LATTICE_EXPERIMENTAL_IGEMM_CONV') != '1':
        return False
    if relation.contract.kind not in ('forward', 'target'):
        return False
    if feats.dtype != mx.float16 or weight.dtype != mx.float16:
        return False
    if relation.n_kernels != 27:
        return False
    if feats.ndim != 2 or int(feats.shape[1]) != 32:
        return False
    if weight.ndim == 3:
        return int(weight.shape[0]) == 27 and tuple(weight.shape[1:]) == (
            32,
            32,
        )
    return weight.ndim == 5 and tuple(weight.shape) == (32, 3, 3, 3, 32)


def _mapped_weight(weight: mx.array) -> mx.array:
    if weight.ndim == 3:
        return weight
    return weight.transpose(1, 2, 3, 4, 0).reshape((-1, 32, 32))


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
