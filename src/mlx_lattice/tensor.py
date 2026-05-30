from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_lattice.point import KernelMap, build_kernel_map
from mlx_lattice.types import Triple, triple


@dataclass(frozen=True)
class SparseTensor:
    coords: mx.array
    feats: mx.array
    stride: Triple = (1, 1, 1)
    _maps: dict[tuple[Triple, Triple], KernelMap] = field(
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
        object.__setattr__(self, 'coords', coords)
        object.__setattr__(self, 'feats', feats)
        object.__setattr__(self, 'stride', triple(stride, name='stride'))
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
        return SparseTensor(
            self.coords if coords is None else coords,
            self.feats if feats is None else feats,
            self.stride if stride is None else stride,
        )

    def kernel_map(
        self,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
    ) -> KernelMap:
        key = (
            triple(kernel_size, name='kernel_size'),
            triple(stride, name='stride'),
        )
        if key not in self._maps:
            self._maps[key] = build_kernel_map(
                self.coords,
                kernel_size=key[0],
                stride=key[1],
            )
        return self._maps[key]
