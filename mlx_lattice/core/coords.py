from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx

from mlx_lattice.core.types import Triple, triple


@dataclass(frozen=True, slots=True)
class CoordinateMapKey:
    id: int
    stride: Triple


@dataclass(slots=True)
class CoordinateManager:
    _next_id: int = 0
    _coords: dict[CoordinateMapKey, mx.array] = field(default_factory=dict)
    _coord_keys: dict[tuple[int, Triple], CoordinateMapKey] = field(
        default_factory=dict
    )

    def insert(
        self,
        coords: mx.array,
        stride: int | Sequence[int] = 1,
    ) -> CoordinateMapKey:
        validate_coords(coords)
        normalized = triple(stride, name='stride')
        cache_key = (id(coords), normalized)
        if cache_key in self._coord_keys:
            return self._coord_keys[cache_key]

        key = CoordinateMapKey(self._next_id, normalized)
        self._next_id += 1
        self._coords[key] = coords
        self._coord_keys[cache_key] = key
        return key

    def coords(self, key: CoordinateMapKey) -> mx.array:
        return self._coords[key]

    def inverse_map(
        self,
        source: CoordinateMapKey,
        target: CoordinateMapKey,
    ) -> mx.array:
        return inverse_map(self.coords(source), self.coords(target))


def validate_coords(coords: mx.array, *, name: str = 'coords') -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def lookup_coords(coords: mx.array, queries: mx.array) -> mx.array:
    validate_coords(coords)
    validate_coords(queries, name='queries')
    if coords.dtype != queries.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')

    rows = {
        _coord_key(row): index for index, row in enumerate(_rows(coords))
    }
    values = [rows.get(_coord_key(row), -1) for row in _rows(queries)]
    return mx.array(values, dtype=mx.int32)


def contains_coords(coords: mx.array, queries: mx.array) -> mx.array:
    return lookup_coords(coords, queries) >= 0


def inverse_map(source: mx.array, target: mx.array) -> mx.array:
    return lookup_coords(source, target)


def same_coords(lhs: mx.array, rhs: mx.array) -> bool:
    validate_coords(lhs, name='lhs')
    validate_coords(rhs, name='rhs')
    return lhs.shape == rhs.shape and _rows(lhs) == _rows(rhs)


def _rows(coords: mx.array) -> list[list[int]]:
    values = cast(list[list[int]], coords.tolist())
    return [[int(item) for item in row] for row in values]


def _coord_key(row: Sequence[int]) -> tuple[int, int, int, int]:
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
