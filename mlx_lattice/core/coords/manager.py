from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count

import mlx.core as mx

from mlx_lattice.core.coords.builders import (
    build_generative_relation,
    build_kernel_relation,
    build_target_kernel_relation,
    build_transposed_kernel_relation,
)
from mlx_lattice.core.coords.set_ops import inverse_map
from mlx_lattice.core.coords.validation import validate_coords
from mlx_lattice.core.relations import KernelRelation, KernelSpec
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
    _identity_keys: dict[tuple[int, int, Triple], CoordinateMapKey] = field(
        default_factory=dict
    )
    _default_identity_keys: dict[tuple[int, Triple], CoordinateMapKey] = (
        field(default_factory=dict)
    )
    _active_rows: dict[CoordinateMapKey, mx.array] = field(
        default_factory=dict
    )
    _kernel_relations: dict[
        tuple[CoordinateMapKey, CoordinateMapKey | None, KernelSpec, str],
        KernelRelation,
    ] = field(default_factory=dict)

    def insert_coords(
        self,
        coords: mx.array,
        stride: int | Sequence[int] = 1,
        active_rows: mx.array | None = None,
    ) -> CoordinateMapKey:
        """Register a coordinate array by object identity and stride."""
        validate_coords(coords)
        normalized = triple(stride, name='stride')
        if active_rows is None:
            default_key = (id(coords), normalized)
            if default_key in self._default_identity_keys:
                return self._default_identity_keys[default_key]
            active = mx.array([coords.shape[0]], dtype=mx.int32)
        else:
            active = active_rows
        cache_key = (id(coords), id(active), normalized)
        if cache_key in self._identity_keys:
            return self._identity_keys[cache_key]

        key = CoordinateMapKey(self._next_id, normalized, self._manager_id)
        self._next_id += 1
        self._coords[key] = coords
        self._active_rows[key] = active
        self._identity_keys[cache_key] = key
        if active_rows is None:
            self._default_identity_keys[(id(coords), normalized)] = key
        return key

    def owns(self, key: CoordinateMapKey) -> bool:
        return key.manager_id == self._manager_id and key in self._coords

    def coords(self, key: CoordinateMapKey) -> mx.array:
        if not self.owns(key):
            raise ValueError(
                'coordinate key does not belong to this manager.'
            )
        return self._coords[key]

    def active_rows(self, key: CoordinateMapKey) -> mx.array:
        if not self.owns(key):
            raise ValueError(
                'coordinate key does not belong to this manager.'
            )
        return self._active_rows[key]

    def inverse_map(
        self,
        source: CoordinateMapKey,
        target: CoordinateMapKey,
    ) -> mx.array:
        return inverse_map(self.coords(source), self.coords(target))

    def kernel_relation(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelRelation:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, None, spec, 'forward')
        if cache_key not in self._kernel_relations:
            self._kernel_relations[cache_key] = build_kernel_relation(
                self.coords(key),
                active_rows=self.active_rows(key),
                kernel_size=spec.size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
        return self._kernel_relations[cache_key]

    def generative_relation(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
    ) -> KernelRelation:
        spec = KernelSpec(size=kernel_size, stride=stride)
        cache_key = (key, None, spec, 'generative')
        if cache_key not in self._kernel_relations:
            self._kernel_relations[cache_key] = build_generative_relation(
                self.coords(key),
                active_rows=self.active_rows(key),
                kernel_size=spec.size,
                stride=spec.stride,
            )
        return self._kernel_relations[cache_key]

    def transposed_kernel_relation(
        self,
        key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 2,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelRelation:
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, None, spec, 'transpose')
        if cache_key not in self._kernel_relations:
            self._kernel_relations[cache_key] = (
                build_transposed_kernel_relation(
                    self.coords(key),
                    active_rows=self.active_rows(key),
                    kernel_size=spec.size,
                    stride=spec.stride,
                    padding=spec.padding,
                    dilation=spec.dilation,
                )
            )
        return self._kernel_relations[cache_key]

    def target_kernel_relation(
        self,
        key: CoordinateMapKey,
        target_key: CoordinateMapKey,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> KernelRelation:
        if not self.owns(key) or not self.owns(target_key):
            raise ValueError(
                'input and target coordinate keys must belong to this manager.'
            )
        spec = KernelSpec(
            size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        cache_key = (key, target_key, spec, 'target')
        if cache_key not in self._kernel_relations:
            self._kernel_relations[cache_key] = (
                build_target_kernel_relation(
                    self.coords(key),
                    self.coords(target_key),
                    active_rows=self.active_rows(key),
                    target_active_rows=self.active_rows(target_key),
                    kernel_size=spec.size,
                    stride=spec.stride,
                    padding=spec.padding,
                    dilation=spec.dilation,
                )
            )
        return self._kernel_relations[cache_key]
