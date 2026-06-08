from __future__ import annotations

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.relations import (
    EdgeCooPlan,
    KernelRelation,
    edge_coo_plan,
)

__all__ = [
    'execute_pool_max',
    'execute_pool_sum',
    'execute_spmm',
    'pool_max_edge_coo',
    'pool_sum_edge_coo',
    'spmm_edge_coo',
]


def execute_spmm(
    feats: mx.array,
    weights: mx.array,
    relation: KernelRelation,
) -> mx.array:
    return spmm_edge_coo(feats, weights, edge_coo_plan(relation))


def execute_pool_sum(feats: mx.array, relation: KernelRelation) -> mx.array:
    return pool_sum_edge_coo(feats, edge_coo_plan(relation))


def execute_pool_max(feats: mx.array, relation: KernelRelation) -> mx.array:
    return pool_max_edge_coo(feats, edge_coo_plan(relation))


def spmm_edge_coo(
    feats: mx.array,
    weights: mx.array,
    plan: EdgeCooPlan,
) -> mx.array:
    return ext.spmm_edges(
        feats,
        weights,
        plan.edge_coo.in_rows,
        plan.edge_coo.out_rows,
        plan.edge_coo.kernel_ids,
        plan.edge_count,
        plan.n_out_rows,
    )


def pool_sum_edge_coo(feats: mx.array, plan: EdgeCooPlan) -> mx.array:
    return ext.pool_sum_edges(
        feats,
        plan.edge_coo.in_rows,
        plan.edge_coo.out_rows,
        plan.edge_count,
        plan.n_out_rows,
    )


def pool_max_edge_coo(feats: mx.array, plan: EdgeCooPlan) -> mx.array:
    return ext.pool_max_edges(
        feats,
        plan.edge_coo.in_rows,
        plan.edge_coo.out_rows,
        plan.edge_count,
        plan.n_out_rows,
    )
