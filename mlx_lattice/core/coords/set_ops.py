from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import (
    validate_coord_pair,
    validate_coords,
)
from mlx_lattice.core.types import Triple, triple


@dataclass(frozen=True, slots=True)
class CoordinateSet:
    """Capacity coordinate buffer with a lazy active count."""

    coords: mx.array
    active_rows: mx.array

    def __post_init__(self) -> None:
        validate_coords(self.coords)
        if (
            self.active_rows.shape != (1,)
            or self.active_rows.dtype != mx.int32
        ):
            raise ValueError(
                'active_rows must have shape (1,) and int32 dtype.'
            )

    @property
    def capacity(self) -> int:
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        return self.active_rows


def downsample_coords(
    coords: mx.array,
    stride: int | Sequence[int] = 2,
) -> CoordinateSet:
    """Downsample coordinates by integer spatial stride."""
    validate_coords(coords)
    step = triple(stride, name='stride')
    _require_positive(step, 'stride')
    return CoordinateSet(*ext.downsample_coords(coords, step))


def union_coords(lhs: mx.array, rhs: mx.array) -> CoordinateSet:
    """Return the coordinate-set union of two coordinate arrays."""
    validate_coord_pair(lhs, rhs)
    return CoordinateSet(*ext.union_coords(lhs, rhs))


def intersection_coords(lhs: mx.array, rhs: mx.array) -> CoordinateSet:
    """Return coordinates present in both input coordinate arrays."""
    validate_coord_pair(lhs, rhs)
    return CoordinateSet(*ext.intersection_coords(lhs, rhs))


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    """Map query coordinates to row indices in ``coords``.

    Missing query rows are encoded as ``-1``.
    """
    validate_coord_pair(coords, queries, rhs_name='queries')
    return ext.lookup_coords(coords, queries)


def contains_coords(coords: mx.array, queries: mx.array) -> mx.array:
    """Return a boolean mask indicating which queries exist in ``coords``."""
    return lookup_coords(coords, queries) >= 0


def inverse_map(source: mx.array, target: mx.array) -> mx.array:
    """Return row indices that gather ``target`` rows from ``source``."""
    return lookup_coords(source, target)


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')
