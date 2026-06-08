from __future__ import annotations

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.maps import KernelMap

__all__ = ['spmm_edges']


def spmm_edges(
    feats: mx.array,
    weights: mx.array,
    mapping: KernelMap,
) -> mx.array:
    if mapping.n_out_rows is None:
        raise ValueError('mapping must define n_out_rows.')
    return ext.spmm_edges(
        feats,
        weights,
        mapping.in_rows,
        mapping.out_rows,
        mapping.kernel_ids,
        mapping.n_out_rows,
    )
