from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count

import mlx.core as mx

from mlx_lattice.core.coords.builders import (
    build_generative_map,
    build_kernel_map,
    build_transposed_kernel_map,
)
from mlx_lattice.core.coords.set_ops import inverse_map
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.maps import KernelMap, KernelSpec
from mlx_lattice.core.types import Triple, triple

_manager_ids = count()


def _next_manager_id() -> int:
    return next(_manager_ids)


@dataclass(frozen=True, slots=True)
class CoordinateMapKey:
    id: int
    stride: Triple
    manager_id: int


@dataclass(slots=True)
class CoordinateManager:
    _manager_id: int = field(default_factory=_next_manager_id, init=False)
    _next_id: int = 0
    _coords: dict[CoordinateMapKey, mx.array] = field(default_factory=dict)
    _identity_keys: dict[tuple[int, Triple], CoordinateMapKey] = field(
        default_factory=dict
    )
    _kernel_maps: dict[
        tuple[CoordinateMapKey, KernelSpec, str], KernelMap
    ] = field(default_factory=dict)

    def insert_coords(
        self,
        coords: mx.array,
        stride: int | Sequence[int] = 1,
    ) -> CoordinateMapKey:
        """Register a coordinate array by object identity and stride."""
        validate_coords(coords)
        normalized = triple(stride, name='stride')
        cache_key = (id(coords), normalized)
        if cache_key in self._identity_keys:
            return self._identity_keys[cache_key]

        key = CoordinateMapKey(self._next_id, normalized, self._manager_id)
        self._next_id += 1
        self._coords[key] = coords
        self._identity_keys[cache_key] = key
        return key

    def owns(self, key: CoordinateMapKey) -> bool:
        return key.manager_id == self._manager_id and key in self._coords

    def coords(self, key: CoordinateMapKey) -> mx.array:
        if not self.owns(key):
            raise ValueError(
                'coordinate key does not belong to this manager.'
            )
        return self._coords[key]

    def inverse_map(
        self,
        source: CoordinateMapKey,
        target: CoordinateMapKey,
    ) -> mx.array:
        return inverse_map(self.coords(source), self.coords(target))

    def kernel_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelMap:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, spec, 'forward')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_kernel_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
        return self._kernel_maps[cache_key]

    def generative_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
    ) -> KernelMap:
        spec = KernelSpec(size=kernel_size, stride=stride)
        cache_key = (key, spec, 'generative')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_generative_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
            )
        return self._kernel_maps[cache_key]

    def transposed_kernel_map(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelMap:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, spec, 'transpose')
        if cache_key not in self._kernel_maps:
            self._kernel_maps[cache_key] = build_transposed_kernel_map(
                self.coords(key),
                kernel_size=spec.size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
        return self._kernel_maps[cache_key]
