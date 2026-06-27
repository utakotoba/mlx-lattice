from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import validate_coord_pair

type SparseJoin = Literal['inner', 'left', 'right', 'outer']


@dataclass(frozen=True, slots=True)
class SparseAlignment:
    """Coordinate value alignment for two sparse tensors."""

    coords: mx.array
    active_rows: mx.array
    lhs_rows: mx.array
    rhs_rows: mx.array

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 4:
            raise ValueError('coords must have shape (N, 4).')
        if self.coords.dtype not in (mx.int32, mx.int64):
            raise ValueError('coords must be int32 or int64.')
        if (
            self.active_rows.shape != (1,)
            or self.active_rows.dtype != mx.int32
        ):
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )
        if self.lhs_rows.shape != (self.coords.shape[0],):
            raise ValueError('lhs_rows must have shape (N,).')
        if self.rhs_rows.shape != (self.coords.shape[0],):
            raise ValueError('rhs_rows must have shape (N,).')
        if (
            self.lhs_rows.dtype != mx.int32
            or self.rhs_rows.dtype != mx.int32
        ):
            raise ValueError('alignment rows must be int32.')

    @property
    def capacity(self) -> int:
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        return self.active_rows


def build_sparse_alignment(
    lhs_coords: mx.array,
    lhs_active_rows: mx.array,
    rhs_coords: mx.array,
    rhs_active_rows: mx.array,
    *,
    join: SparseJoin = 'inner',
) -> SparseAlignment:
    validate_coord_pair(lhs_coords, rhs_coords)
    if lhs_coords.dtype != mx.int32:
        raise ValueError(
            'sparse alignment currently requires int32 coords.'
        )
    _validate_active_rows(lhs_active_rows, 'lhs_active_rows')
    _validate_active_rows(rhs_active_rows, 'rhs_active_rows')
    return SparseAlignment(
        *ext.build_sparse_alignment(
            lhs_coords,
            lhs_active_rows,
            rhs_coords,
            rhs_active_rows,
            _validate_join(join),
        )
    )


def _validate_join(value: str) -> SparseJoin:
    if value == 'inner':
        return 'inner'
    if value == 'left':
        return 'left'
    if value == 'right':
        return 'right'
    if value == 'outer':
        return 'outer'
    raise ValueError("join must be 'inner', 'left', 'right', or 'outer'.")


def _validate_active_rows(value: mx.array, name: str) -> None:
    if value.shape != (1,) or value.dtype != mx.int32:
        raise ValueError(f'{name} must have shape (1,) and int32 dtype.')
