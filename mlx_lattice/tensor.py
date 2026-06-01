from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx

from mlx_lattice.coords import CoordinateManager, CoordinateMapKey
from mlx_lattice.point import KernelMap, contains_coords, inverse_map
from mlx_lattice.point import lookup_coords as lookup_coord_rows
from mlx_lattice.types import Triple, triple


@dataclass(frozen=True)
class SparseTensor:
    coords: mx.array
    feats: mx.array
    stride: Triple = (1, 1, 1)
    coord_key: CoordinateMapKey | None = None
    coord_manager: CoordinateManager | None = None
    batch_counts: tuple[int, ...] | None = None
    _maps: dict[tuple[Triple, Triple, Triple, Triple], KernelMap] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        coords: mx.array,
        feats: mx.array,
        stride: int | Sequence[int] = 1,
        *,
        coord_key: CoordinateMapKey | None = None,
        coord_manager: CoordinateManager | None = None,
        batch_counts: Sequence[int] | None = None,
    ) -> None:
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError('coords must have shape (N, 4).')
        if feats.ndim != 2:
            raise ValueError('feats must have shape (N, C).')
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(
                'coords and feats must have the same row count.'
            )
        if coords.dtype not in (mx.int32, mx.int64):
            raise ValueError('coords must be int32 or int64.')
        normalized_stride = triple(stride, name='stride')
        manager = (
            CoordinateManager() if coord_manager is None else coord_manager
        )
        key = (
            manager.insert(coords, normalized_stride)
            if coord_key is None
            else coord_key
        )
        object.__setattr__(self, 'coords', coords)
        object.__setattr__(self, 'feats', feats)
        object.__setattr__(self, 'stride', normalized_stride)
        object.__setattr__(self, 'coord_key', key)
        object.__setattr__(self, 'coord_manager', manager)
        object.__setattr__(
            self,
            'batch_counts',
            None
            if batch_counts is None
            else tuple(int(v) for v in batch_counts),
        )
        object.__setattr__(self, '_maps', {})

    @property
    def n_points(self) -> int:
        return int(self.coords.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_points, self.channels)

    @property
    def dtype(self) -> mx.Dtype:
        return self.feats.dtype

    @property
    def channels(self) -> int:
        return int(self.feats.shape[1])

    def astype(self, dtype: mx.Dtype) -> SparseTensor:
        return self.replace(feats=self.feats.astype(dtype))

    def replace(
        self,
        *,
        coords: mx.array | None = None,
        feats: mx.array | None = None,
        stride: int | Sequence[int] | None = None,
    ) -> SparseTensor:
        same_coords = coords is None or coords is self.coords
        return SparseTensor(
            self.coords if coords is None else coords,
            self.feats if feats is None else feats,
            self.stride if stride is None else stride,
            coord_key=self.coord_key if same_coords else None,
            coord_manager=self.coord_manager if same_coords else None,
            batch_counts=self.batch_counts if same_coords else None,
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
        )

    def kernel_map(
        self,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelMap:
        key = (
            triple(kernel_size, name='kernel_size'),
            triple(stride, name='stride'),
            triple(padding, name='padding'),
            triple(dilation, name='dilation'),
        )
        if key not in self._maps:
            if self.coord_key is None or self.coord_manager is None:
                raise RuntimeError(
                    'SparseTensor has no coordinate manager.'
                )
            self._maps[key] = self.coord_manager.kernel_map(
                self.coord_key,
                kernel_size=key[0],
                stride=key[1],
                padding=key[2],
                dilation=key[3],
            )
        return self._maps[key]

    @property
    def batch_indices(self) -> mx.array:
        return self.coords[:, 0]

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

    @property
    def batch_rows(self) -> tuple[mx.array, ...]:
        rows: dict[int, list[int]] = {}
        coords = cast(list[list[int]], self.coords.tolist())
        for row, coord in enumerate(coords):
            rows.setdefault(int(coord[0]), []).append(row)
        return tuple(
            mx.array(values, dtype=mx.int32)
            for _, values in sorted(rows.items())
        )

    def same_coords(self, other: SparseTensor) -> bool:
        return (
            self.coord_key is not None
            and self.coord_key == other.coord_key
            and self.coord_manager is other.coord_manager
        ) or (
            self.stride == other.stride
            and self.coords.shape == other.coords.shape
            and self.coords.tolist() == other.coords.tolist()
        )

    def inverse_map(self, other: SparseTensor) -> mx.array:
        if (
            self.coord_key is not None
            and other.coord_key is not None
            and self.coord_manager is not None
            and self.coord_manager is other.coord_manager
        ):
            return self.coord_manager.inverse_map(
                self.coord_key, other.coord_key
            )
        return inverse_map(self.coords, other.coords)

    def lookup_coords(self, queries: mx.array) -> mx.array:
        return lookup_coord_rows(self.coords, queries)

    def contains_coords(self, queries: mx.array) -> mx.array:
        return contains_coords(self.coords, queries)

    def __add__(self, other: SparseTensor) -> SparseTensor:
        if not self.same_coords(other):
            raise ValueError('sparse tensor coordinates must match.')
        return self.replace(feats=self.feats + other.feats)
