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
    """Opaque coordinate identity key owned by a ``CoordinateManager``.

    Keys are lightweight metadata objects. A key is valid only for the manager
    whose ``manager_id`` it stores. The ``stride`` field is part of coordinate
    identity because the same integer rows represent different lattice cells at
    different spatial strides.
    """

    id: int
    stride: Triple
    manager_id: int


@dataclass(slots=True)
class CoordinateManager:
    """Owns coordinate arrays and caches sparse relations by identity.

    Managers are normally created by ``SparseTensor``. Reusing a manager and
    key lets convolutions, pooling, and aligned sparse algebra reuse native
    relation metadata instead of rebuilding it for every call. Cache entries
    are keyed by coordinate identity, optional target identity, kernel geometry,
    and relation kind.
    """

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
        """Register a coordinate array by object identity and stride.

        Args:
            coords: Integer coordinates with shape ``(N, 4)``.
            stride: Spatial lattice stride for the coordinate rows.
            active_rows: Optional ``int32`` scalar ``(1,)`` active-row count.

        Returns:
            A manager-owned key. Re-inserting the same coordinate object,
            active-row object, and stride returns the existing key.
        """
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
        """Return whether ``key`` belongs to this manager."""
        return key.manager_id == self._manager_id and key in self._coords

    def coords(self, key: CoordinateMapKey) -> mx.array:
        """Return the coordinate array registered for ``key``."""
        if not self.owns(key):
            raise ValueError(
                'coordinate key does not belong to this manager.'
            )
        return self._coords[key]

    def active_rows(self, key: CoordinateMapKey) -> mx.array:
        """Return the active-row scalar registered for ``key``."""
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
        """Return row indices that gather ``target`` coordinates from source.

        Missing target rows are encoded by the native set-operation contract.
        Both keys must belong to this manager.
        """
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
        """Build or reuse a forward sparse kernel relation.

        The relation maps input rows to output rows generated by applying
        ``kernel_size``, ``stride``, ``padding``, and ``dilation`` to the input
        coordinate support. The resulting relation is cached for the exact
        normalized geometry.
        """
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
        """Build or reuse a generative transpose-convolution relation.

        Generative relations create output support from each input coordinate
        and transpose-convolution stride without requiring an existing target
        coordinate set.
        """
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
        """Build or reuse a transpose-convolution relation.

        The relation represents the sparse transpose of the forward
        convolution geometry and carries output coordinates plus CSR views for
        native execution.
        """
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
        """Build or reuse a relation from one key to explicit target coords.

        ``target_key`` fixes the output support. Only input rows that connect
        to those target rows through the kernel geometry contribute edges.
        """
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
