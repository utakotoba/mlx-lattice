from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_lattice._native import native
from mlx_lattice.core.coords.validation import validate_coords


@dataclass(frozen=True, slots=True)
class SparseOccupancy:
    coords: mx.array
    active_rows: mx.array
    occupancy: mx.array

    def __post_init__(self) -> None:
        validate_coords(self.coords)
        _validate_active_rows(self.active_rows)
        if (
            self.occupancy.ndim != 1
            or self.occupancy.dtype != mx.int32
            or self.occupancy.shape[0] != self.coords.shape[0]
        ):
            raise ValueError(
                'occupancy must have shape (N,) and int32 dtype.'
            )

    @property
    def capacity(self) -> int:
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        return self.active_rows


@dataclass(frozen=True, slots=True)
class OccupancyExpansion:
    coords: mx.array
    active_rows: mx.array
    parent_rows: mx.array
    child_indices: mx.array

    def __post_init__(self) -> None:
        validate_coords(self.coords)
        _validate_active_rows(self.active_rows)
        _validate_row_array(self.parent_rows, 'parent_rows')
        _validate_row_array(self.child_indices, 'child_indices')
        if self.parent_rows.shape[0] != self.coords.shape[0]:
            raise ValueError('parent_rows must have shape (N,).')
        if self.child_indices.shape[0] != self.coords.shape[0]:
            raise ValueError('child_indices must have shape (N,).')

    @property
    def capacity(self) -> int:
        return int(self.coords.shape[0])

    @property
    def active_count(self) -> mx.array:
        return self.active_rows


def occupancy_downsample(
    coords: mx.array,
    active_rows: mx.array | None = None,
) -> SparseOccupancy:
    coords = _coords_for_native(coords)
    active_rows = _active_rows_for(coords, active_rows)
    return SparseOccupancy(
        *native.occupancy_downsample(coords, active_rows)
    )


def occupancy_expand(
    coords: mx.array,
    occupancy: mx.array,
    active_rows: mx.array | None = None,
) -> OccupancyExpansion:
    coords = _coords_for_native(coords)
    active_rows = _active_rows_for(coords, active_rows)
    _validate_row_array(occupancy, 'occupancy')
    if occupancy.shape[0] != coords.shape[0]:
        raise ValueError('occupancy must have shape (N,).')
    return OccupancyExpansion(
        *native.occupancy_expand(coords, active_rows, occupancy)
    )


def child_coords_from_indices(
    parent_coords: mx.array,
    child_indices: mx.array,
) -> mx.array:
    parent_coords = _coords_for_native(parent_coords)
    _validate_row_array(child_indices, 'child_indices')
    if child_indices.shape[0] != parent_coords.shape[0]:
        raise ValueError('child_indices must have shape (N,).')
    return native.child_coords_from_indices(parent_coords, child_indices)


def _coords_for_native(coords: mx.array) -> mx.array:
    validate_coords(coords)
    if coords.dtype == mx.int32:
        return coords
    return coords.astype(mx.int32)


def _active_rows_for(
    coords: mx.array,
    active_rows: mx.array | None,
) -> mx.array:
    if active_rows is None:
        return mx.array([coords.shape[0]], dtype=mx.int32)
    _validate_active_rows(active_rows)
    return active_rows


def _validate_active_rows(active_rows: mx.array) -> None:
    if active_rows.shape != (1,) or active_rows.dtype != mx.int32:
        raise ValueError(
            'active_rows must have shape (1,) and int32 dtype.'
        )


def _validate_row_array(value: mx.array, name: str) -> None:
    if value.ndim != 1 or value.dtype != mx.int32:
        raise ValueError(f'{name} must have shape (N,) and int32 dtype.')
