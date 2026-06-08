from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice.core.types import Triple


@dataclass(frozen=True, slots=True)
class EdgeCoo:
    """Concrete edge-COO view of a logical sparse neighborhood relation."""

    in_rows: mx.array
    out_rows: mx.array
    kernel_ids: mx.array

    def __post_init__(self) -> None:
        _validate_row_array(self.in_rows, name='in_rows')
        _validate_row_array(self.out_rows, name='out_rows')
        _validate_row_array(self.kernel_ids, name='kernel_ids')
        _require_same_rows(
            self.in_rows,
            self.out_rows,
            self.kernel_ids,
            names=('in_rows', 'out_rows', 'kernel_ids'),
        )

    @property
    def n_edges(self) -> int:
        return int(self.in_rows.shape[0])


@dataclass(frozen=True, slots=True, init=False)
class KernelRelation:
    edge_coo: EdgeCoo
    counts: mx.array
    kernel_offsets: tuple[Triple, ...]
    out_coords: mx.array | None = None
    n_in_rows: int | None = None
    n_out_rows: int | None = None
    n_kernels: int | None = None

    def __init__(
        self,
        in_rows: mx.array,
        out_rows: mx.array,
        kernel_ids: mx.array,
        *,
        counts: mx.array | None = None,
        kernel_offsets: tuple[Triple, ...] = (),
        out_coords: mx.array | None = None,
        n_in_rows: int | None = None,
        n_out_rows: int | None = None,
        n_kernels: int | None = None,
    ) -> None:
        if out_coords is not None:
            _validate_coords(out_coords, name='out_coords')
        if counts is None:
            counts = mx.array(
                [
                    in_rows.shape[0],
                    0 if out_coords is None else out_coords.shape[0],
                ],
                dtype=mx.int32,
            )
        _validate_counts(counts)

        edge_coo = EdgeCoo(in_rows, out_rows, kernel_ids)
        normalized_kernel_offsets = tuple(
            (int(x), int(y), int(z)) for x, y, z in kernel_offsets
        )
        normalized_n_in_rows = _optional_count(n_in_rows, 'n_in_rows')
        normalized_n_out_rows = _optional_count(n_out_rows, 'n_out_rows')
        normalized_n_kernels = _optional_count(n_kernels, 'n_kernels')
        if (
            normalized_kernel_offsets
            and normalized_n_kernels is not None
            and len(normalized_kernel_offsets) != normalized_n_kernels
        ):
            raise ValueError('n_kernels must match kernel_offsets.')
        if normalized_kernel_offsets:
            normalized_n_kernels = len(normalized_kernel_offsets)
        if out_coords is not None:
            out_coord_rows = int(out_coords.shape[0])
            if (
                normalized_n_out_rows is not None
                and normalized_n_out_rows != out_coord_rows
            ):
                raise ValueError('n_out_rows must match out_coords rows.')
            normalized_n_out_rows = out_coord_rows

        object.__setattr__(self, 'edge_coo', edge_coo)
        object.__setattr__(self, 'counts', counts)
        object.__setattr__(
            self, 'kernel_offsets', normalized_kernel_offsets
        )
        object.__setattr__(self, 'out_coords', out_coords)
        object.__setattr__(self, 'n_in_rows', normalized_n_in_rows)
        object.__setattr__(self, 'n_out_rows', normalized_n_out_rows)
        object.__setattr__(self, 'n_kernels', normalized_n_kernels)

    @property
    def n_edges(self) -> int:
        return self.edge_coo.n_edges

    @property
    def edge_count(self) -> mx.array:
        return self.counts[:1]

    @property
    def out_count(self) -> mx.array:
        return self.counts[1:2]


@dataclass(frozen=True, slots=True)
class EdgeCooPlan:
    """Current baseline execution plan lowered from a kernel relation.

    The plan is intentionally internal-facing: operators ask for a plan, not a
    sparse format. Future native lowerings can return CSR, kernel-bucketed, or
    implicit-GEMM plans without changing tensor/module semantics.
    """

    edge_coo: EdgeCoo
    n_out_rows: int
    edge_count: mx.array


def edge_coo_plan(relation: KernelRelation) -> EdgeCooPlan:
    if relation.n_out_rows is None:
        raise ValueError('kernel relation must define n_out_rows.')
    return EdgeCooPlan(
        relation.edge_coo, relation.n_out_rows, relation.edge_count
    )


# MARK: - helpers


def _validate_row_array(value: mx.array, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f'{name} must have shape (E,).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_coords(value: mx.array, *, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_counts(value: mx.array) -> None:
    if value.shape != (2,) or value.dtype != mx.int32:
        raise ValueError(
            'relation counts must have shape (2,) and int32 dtype.'
        )


def _require_same_rows(
    first: mx.array,
    *rest: mx.array,
    names: tuple[str, ...],
) -> None:
    rows = int(first.shape[0])
    for name, value in zip(names[1:], rest, strict=True):
        if int(value.shape[0]) != rows:
            raise ValueError(
                f'{names[0]} and {name} must have the same row count.'
            )


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f'{name} must be non-negative.')
    return normalized
