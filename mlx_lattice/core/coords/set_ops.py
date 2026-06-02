from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core.coords.validation import (
    Coord,
    coord_rows,
    validate_coord_pair,
    validate_coords,
)
from mlx_lattice.core.types import Triple, triple


def downsample_coords(
    coords: mx.array,
    stride: int | Sequence[int] = 2,
) -> mx.array:
    validate_coords(coords)
    step = triple(stride, name='stride')
    _require_positive(step, 'stride')
    return ext.downsample_coords(coords, step)


def downsample_values(
    values: Sequence[Coord],
    stride: Triple,
) -> list[Coord]:
    seen: set[Coord] = set()
    out: list[Coord] = []
    for coord in values:
        quantized = (
            coord[0],
            coord[1] // stride[0],
            coord[2] // stride[1],
            coord[3] // stride[2],
        )
        if quantized not in seen:
            seen.add(quantized)
            out.append(quantized)
    return out


def union_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    validate_coord_pair(lhs, rhs)
    return ext.union_coords(lhs, rhs)


def intersection_coords(lhs: mx.array, rhs: mx.array) -> mx.array:
    validate_coord_pair(lhs, rhs)
    return ext.intersection_coords(lhs, rhs)


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    validate_coord_pair(coords, queries, rhs_name='queries')
    return ext.lookup_coords(coords, queries)


def contains_coords(coords: mx.array, queries: mx.array) -> mx.array:
    return lookup_coords(coords, queries) >= 0


def inverse_map(source: mx.array, target: mx.array) -> mx.array:
    return lookup_coords(source, target)


def same_coords(lhs: mx.array, rhs: mx.array) -> bool:
    validate_coords(lhs, name='lhs')
    validate_coords(rhs, name='rhs')
    return lhs.shape == rhs.shape and coord_rows(lhs) == coord_rows(rhs)


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')
