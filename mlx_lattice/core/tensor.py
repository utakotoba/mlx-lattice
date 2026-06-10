from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice.core.coords import (
    CoordinateManager,
    CoordinateMapKey,
    validate_coords,
)
from mlx_lattice.core.types import Triple, triple


@dataclass(frozen=True, slots=True, init=False)
class SparseTensor:
    coords: mx.array
    feats: mx.array
    stride: Triple
    coord_key: CoordinateMapKey
    coord_manager: CoordinateManager
    batch_counts: tuple[int, ...] | None
    active_rows: mx.array

    def __init__(
        self,
        coords: mx.array,
        feats: mx.array,
        stride: int | Sequence[int] = 1,
        *,
        coord_key: CoordinateMapKey | None = None,
        coord_manager: CoordinateManager | None = None,
        batch_counts: Sequence[int] | None = None,
        active_rows: mx.array | None = None,
    ) -> None:
        normalized_stride = triple(stride, name='stride')
        manager, key, owned_coords, normalized_active = (
            _resolve_coordinate_identity(
                coords,
                normalized_stride,
                active_rows,
                coord_key=coord_key,
                coord_manager=coord_manager,
            )
        )
        if feats.ndim != 2:
            raise ValueError('feats must have shape (N, C).')
        if owned_coords.shape[0] != feats.shape[0]:
            raise ValueError(
                'coords and feats must have the same row count.'
            )

        normalized_counts = _batch_counts(
            batch_counts, rows=owned_coords.shape[0]
        )

        object.__setattr__(self, 'coords', owned_coords)
        object.__setattr__(self, 'feats', feats)
        object.__setattr__(self, 'stride', normalized_stride)
        object.__setattr__(self, 'coord_key', key)
        object.__setattr__(self, 'coord_manager', manager)
        object.__setattr__(self, 'batch_counts', normalized_counts)
        object.__setattr__(self, 'active_rows', normalized_active)

    @property
    def capacity(self) -> int:
        """Static row capacity of the sparse buffers."""
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        """Lazy MLX scalar array containing the number of active sparse rows."""
        return self.active_rows

    @property
    def channels(self) -> int:
        return int(self.feats.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.capacity, self.channels)

    @property
    def dtype(self) -> mx.Dtype:
        return self.feats.dtype

    @property
    def batch_indices(self) -> mx.array:
        return self.coords[:, 0]

    @property
    def batch_rows(self) -> tuple[mx.array, ...]:
        counts = self._require_batch_counts()
        start = 0
        batches = []
        for count in counts:
            stop = start + count
            batches.append(mx.arange(start, stop, dtype=mx.int32))
            start = stop
        return tuple(batches)

    @property
    def decomposed_coordinates(self) -> tuple[mx.array, ...]:
        return tuple(
            mx.take(self.coords[:, 1:], rows, axis=0)
            for rows in self.batch_rows
        )

    @property
    def decomposed_features(self) -> tuple[mx.array, ...]:
        return tuple(
            mx.take(self.feats, rows, axis=0) for rows in self.batch_rows
        )

    def astype(self, dtype: mx.Dtype) -> SparseTensor:
        return self.replace(feats=self.feats.astype(dtype))

    def replace(
        self,
        *,
        coords: mx.array | None = None,
        feats: mx.array | None = None,
        stride: int | Sequence[int] | None = None,
    ) -> SparseTensor:
        next_coords = self.coords if coords is None else coords
        next_stride = (
            self.stride if stride is None else triple(stride, name='stride')
        )
        same_identity = next_coords is self.coords
        reuse_key = same_identity and next_stride == self.stride
        return SparseTensor(
            next_coords,
            self.feats if feats is None else feats,
            next_stride,
            coord_key=self.coord_key if reuse_key else None,
            coord_manager=self.coord_manager,
            batch_counts=self.batch_counts if reuse_key else None,
            active_rows=self.active_rows if reuse_key else None,
        )

    def reuse_coords_from(self, other: SparseTensor) -> SparseTensor:
        if not self.same_coords(other):
            raise ValueError('sparse tensor coordinates must match.')
        return SparseTensor(
            other.coords,
            self.feats,
            other.stride,
            coord_key=other.coord_key,
            coord_manager=other.coord_manager,
            batch_counts=other.batch_counts,
            active_rows=other.active_rows,
        )

    def same_coords(self, other: SparseTensor) -> bool:
        return (
            self.coord_manager is other.coord_manager
            and self.coord_key == other.coord_key
        )

    def __add__(self, other: SparseTensor) -> SparseTensor:
        if not self.same_coords(other):
            raise ValueError('sparse tensor coordinates must match.')
        return self.replace(feats=self.feats + other.feats)

    def _require_batch_counts(self) -> tuple[int, ...]:
        if self.batch_counts is None:
            raise ValueError(
                'batch_counts metadata is required for batch-partitioned '
                'sparse tensor operations.'
            )
        return self.batch_counts


def _resolve_coordinate_identity(
    coords: mx.array,
    stride: Triple,
    active_rows: mx.array | None,
    *,
    coord_key: CoordinateMapKey | None,
    coord_manager: CoordinateManager | None,
) -> tuple[CoordinateManager, CoordinateMapKey, mx.array, mx.array]:
    validate_coords(coords)
    if coord_key is None:
        manager = (
            CoordinateManager() if coord_manager is None else coord_manager
        )
        key = manager.insert_coords(coords, stride, active_rows)
        return (manager, key, coords, manager.active_rows(key))

    if coord_manager is None:
        raise ValueError('coord_manager is required when coord_key is set.')
    if not coord_manager.owns(coord_key):
        raise ValueError('coord_key does not belong to coord_manager.')
    if coord_key.stride != stride:
        raise ValueError('stride must match coord_key stride.')

    owned_coords = coord_manager.coords(coord_key)
    if coords is not owned_coords:
        raise ValueError(
            'coords must be the manager-owned array for coord_key.'
        )
    return (
        coord_manager,
        coord_key,
        owned_coords,
        coord_manager.active_rows(coord_key)
        if active_rows is None
        else _active_rows(active_rows, coords.shape[0]),
    )


def _batch_counts(
    values: Sequence[int] | None,
    *,
    rows: int,
) -> tuple[int, ...] | None:
    if values is None:
        return None

    counts = tuple(int(value) for value in values)
    if any(value < 0 for value in counts):
        raise ValueError('batch_counts must be non-negative.')
    if sum(counts) != rows:
        raise ValueError('batch_counts must cover all sparse rows.')
    return counts


def _active_rows(value: mx.array | None, capacity: int) -> mx.array:
    if value is None:
        return mx.array([capacity], dtype=mx.int32)
    if value.shape != (1,) or value.dtype != mx.int32:
        raise ValueError(
            'active_rows must have shape (1,) and int32 dtype.'
        )
    return value
