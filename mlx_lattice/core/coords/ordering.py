from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice._native import native
from mlx_lattice.core.coords.validation import validate_coords


@dataclass(frozen=True, slots=True)
class CoordinateOrdering:
    coords: mx.array
    order: mx.array
    inverse_rows: mx.array

    def __post_init__(self) -> None:
        validate_coords(self.coords)
        if (
            self.order.ndim != 1
            or self.order.dtype != mx.int32
            or self.order.shape[0] != self.coords.shape[0]
        ):
            raise ValueError('order must have shape (N,) and int32 dtype.')
        if (
            self.inverse_rows.ndim != 1
            or self.inverse_rows.dtype != mx.int32
            or self.inverse_rows.shape[0] != self.coords.shape[0]
        ):
            raise ValueError(
                'inverse_rows must have shape (N,) and int32 dtype.'
            )


def morton_codes(coords: mx.array) -> mx.array:
    validate_coords(coords)
    native_coords = (
        coords if coords.dtype == mx.int32 else coords.astype(mx.int32)
    )
    return native.morton_codes(native_coords)


def morton_order(coords: mx.array) -> mx.array:
    return mx.argsort(morton_codes(coords)).astype(mx.int32)


def morton_sort_coords(coords: mx.array) -> CoordinateOrdering:
    order = morton_order(coords)
    sorted_coords = mx.take(coords, order, axis=0)
    inverse = mx.argsort(order).astype(mx.int32)
    return CoordinateOrdering(sorted_coords, order, inverse)
