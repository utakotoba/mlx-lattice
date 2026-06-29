from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice.core.coords.builders import (
    build_generative_relation,
    build_kernel_relation,
    build_knn_relation,
    build_radius_relation,
    build_target_kernel_relation,
    build_transposed_kernel_relation,
    kernel_offsets,
)
from mlx_lattice.core.relations import KernelRelation, NeighborRelation
from mlx_lattice.core.tensor import SparseTensor

__all__ = [
    'build_generative_relation',
    'build_kernel_relation',
    'build_knn_relation',
    'build_radius_relation',
    'build_target_kernel_relation',
    'build_transposed_kernel_relation',
    'gather_neighbor_features',
    'generative_kernel_relation',
    'kernel_offsets',
    'kernel_relation',
    'knn_relation',
    'radius_relation',
    'target_kernel_relation',
    'transposed_kernel_relation',
]


def kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build the forward kernel relation for a sparse tensor.

    This is the functional wrapper around ``x.coord_manager.kernel_relation``.
    The returned relation can be inspected or passed to lower-level execution
    helpers.
    """
    return x.coord_manager.kernel_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def generative_kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> KernelRelation:
    """Build the generative transpose-convolution relation for a tensor."""
    return x.coord_manager.generative_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
    )


def transposed_kernel_relation(
    x: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build the transpose-convolution relation for a sparse tensor."""
    return x.coord_manager.transposed_kernel_relation(
        x.coord_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def target_kernel_relation(
    x: SparseTensor,
    target: SparseTensor,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> KernelRelation:
    """Build a relation from ``x`` to an explicit target sparse tensor.

    Target coordinates fix the output support. If ``target`` belongs to a
    different coordinate manager, its coordinate array is inserted into
    ``x``'s manager before the relation is cached.
    """
    if x.coord_manager is not target.coord_manager:
        target_key = x.coord_manager.insert_coords(
            target.coords, target.stride, target.active_rows
        )
    else:
        target_key = target.coord_key
    return x.coord_manager.target_kernel_relation(
        x.coord_key,
        target_key,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def knn_relation(
    source: SparseTensor,
    query: SparseTensor | None = None,
    *,
    k: int,
) -> NeighborRelation:
    """Build a k-nearest-neighbor relation for sparse tensors.

    ``source`` supplies candidate rows and ``query`` supplies query rows. When
    ``query`` is omitted, the relation is built within ``source``. Both tensors
    must use the same sparse stride.
    """
    query = source if query is None else query
    _require_matching_stride(source, query)
    return build_knn_relation(
        source.coords,
        query.coords,
        source_active_rows=source.active_rows,
        query_active_rows=query.active_rows,
        k=k,
    )


def radius_relation(
    source: SparseTensor,
    query: SparseTensor | None = None,
    *,
    radius: float,
    max_neighbors: int | None = None,
) -> NeighborRelation:
    """Build a radius-neighborhood relation for sparse tensors.

    ``max_neighbors`` optionally caps the number of source rows associated with
    each query row. Distances are stored in the returned relation in edge order.
    """
    query = source if query is None else query
    _require_matching_stride(source, query)
    return build_radius_relation(
        source.coords,
        query.coords,
        source_active_rows=source.active_rows,
        query_active_rows=query.active_rows,
        radius=radius,
        max_neighbors=max_neighbors,
    )


def gather_neighbor_features(
    source: SparseTensor,
    relation: NeighborRelation,
) -> mx.array:
    """Gather source features in neighbor-edge order.

    The result has shape ``(E, C)`` where ``E`` is the neighbor edge capacity
    and ``C`` is ``source.channels``.
    """
    rows = relation.edges.source_rows.astype(mx.int32)
    return mx.take(source.feats, rows, axis=0)


def _require_matching_stride(
    source: SparseTensor, query: SparseTensor
) -> None:
    if source.stride != query.stride:
        raise ValueError(
            'source and query tensors must use the same coordinate stride.'
        )
