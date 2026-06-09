from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx_lattice.core import SparseTensor


@dataclass(frozen=True, slots=True)
class SparseArrays:
    coords: mx.array
    feats: mx.array
    batch_counts: tuple[int, ...]

    def tensor(self, *, stride: int | Sequence[int] = 1) -> SparseTensor:
        return SparseTensor(
            self.coords,
            self.feats,
            stride=stride,
            batch_counts=self.batch_counts,
        )


@dataclass(frozen=True, slots=True)
class PointArrays:
    points: mx.array
    feats: mx.array
    batch_indices: mx.array


def sparse_arrays(
    *,
    rows: int,
    channels: int,
    batches: int = 1,
) -> SparseArrays:
    batch_counts = _batch_counts(rows, batches)
    coords = []
    feats = []
    row = 0
    for batch, count in enumerate(batch_counts):
        for local in range(count):
            coords.append(_coord(batch, local))
            feats.append(_feature_row(row, channels))
            row += 1
    return SparseArrays(
        coords=mx.array(coords, dtype=mx.int32),
        feats=mx.array(feats, dtype=mx.float32),
        batch_counts=batch_counts,
    )


def point_arrays(
    *,
    rows: int,
    channels: int,
    batches: int = 1,
    duplicate_group: int = 3,
) -> PointArrays:
    batch_counts = _batch_counts(rows, batches)
    points = []
    feats = []
    batch_indices = []
    row = 0
    for batch, count in enumerate(batch_counts):
        for local in range(count):
            bucket = local // max(duplicate_group, 1)
            offset = (local % max(duplicate_group, 1)) * 0.1
            points.append(
                [
                    float(bucket) + offset,
                    float((bucket // 17) % 17) + 0.05,
                    float((bucket // 289) % 17) + 0.025,
                ]
            )
            feats.append(_feature_row(row, channels))
            batch_indices.append(batch)
            row += 1
    return PointArrays(
        points=mx.array(points, dtype=mx.float32),
        feats=mx.array(feats, dtype=mx.float32),
        batch_indices=mx.array(batch_indices, dtype=mx.int32),
    )


def dense_weight(shape: Sequence[int]) -> mx.array:
    total = 1
    for dim in shape:
        total *= int(dim)
    values = [((index % 23) - 11) / 23.0 for index in range(total)]
    return mx.array(values, dtype=mx.float32).reshape(tuple(shape))


def dense_bias(channels: int) -> mx.array:
    return mx.array(
        [((index % 7) - 3) / 19.0 for index in range(channels)],
        dtype=mx.float32,
    )


def _batch_counts(rows: int, batches: int) -> tuple[int, ...]:
    base = rows // batches
    rem = rows % batches
    return tuple(
        base + (1 if batch < rem else 0) for batch in range(batches)
    )


def _coord(batch: int, local: int) -> list[int]:
    return [
        batch,
        local % 97,
        (local // 97) % 97,
        (local // (97 * 97)) % 97,
    ]


def _feature_row(row: int, channels: int) -> list[float]:
    return [
        (((row + 1) * (channel + 3)) % 37) / 37.0
        for channel in range(channels)
    ]
