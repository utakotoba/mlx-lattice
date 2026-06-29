from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import mlx.nn as mxnn
from lattice_contract import IRValueType

ModuleT = TypeVar('ModuleT', bound=type[mxnn.Module])
_EXPORT_SPECS: dict[type[mxnn.Module], ModuleExportSpec] = {}


@dataclass(frozen=True, slots=True)
class ExportAttribute:
    """One manifest attribute exported from an NN module instance."""

    name: str
    getter: Callable[[mxnn.Module], Any]


@dataclass(frozen=True, slots=True)
class ModuleExportSpec:
    """Serializable sparse-module contract attached to an NN class."""

    op: str
    parameters: tuple[str, ...] = ()
    attributes: tuple[ExportAttribute, ...] = ()

    def attribute_values(self, module: mxnn.Module) -> dict[str, Any]:
        """Extract manifest attributes from ``module``."""

        return {
            attribute.name: attribute.getter(module)
            for attribute in self.attributes
        }


def lattice_module(
    op: str,
    *,
    parameters: Sequence[str] = (),
    attributes: Sequence[ExportAttribute] = (),
) -> Callable[[ModuleT], ModuleT]:
    """Attach lattice export metadata to a sparse NN module class."""

    spec = ModuleExportSpec(
        op=op,
        parameters=tuple(parameters),
        attributes=tuple(attributes),
    )

    def decorator(cls: ModuleT) -> ModuleT:
        _EXPORT_SPECS[cls] = spec
        return cls

    return decorator


def module_export_spec(cls: type[mxnn.Module]) -> ModuleExportSpec | None:
    """Return lattice export metadata attached by :func:`lattice_module`."""

    return _EXPORT_SPECS.get(cls)


def path_attribute(name: str, path: str) -> ExportAttribute:
    """Export ``name`` by reading a dotted attribute path from a module."""

    parts = tuple(path.split('.'))

    def getter(module: mxnn.Module) -> Any:
        value: Any = module
        for part in parts:
            value = getattr(value, part)
        return value

    return ExportAttribute(name, getter)


def computed_attribute(
    name: str,
    getter: Callable[[mxnn.Module], Any],
) -> ExportAttribute:
    """Export ``name`` with a module-specific computed getter."""

    return ExportAttribute(name, getter)


def kernel_spec_attributes(*names: str) -> tuple[ExportAttribute, ...]:
    """Export selected :class:`KernelSpec` fields with manifest names."""

    paths = {
        'kernel_size': 'spec.size',
        'stride': 'spec.stride',
        'padding': 'spec.padding',
        'dilation': 'spec.dilation',
    }
    return tuple(path_attribute(name, paths[name]) for name in names)


def annotation_value_type(annotation: object) -> IRValueType:
    """Infer the nearest lattice IR value type from a Python annotation."""

    if annotation is inspect.Signature.empty:
        return 'any'
    text = _annotation_text(annotation)
    candidates = {
        value_type
        for marker, value_type in (
            ('SparseTensor', 'sparse_tensor'),
            ('mx.array', 'dense_tensor'),
            ('KernelRelation', 'relation'),
            ('NeighborRelation', 'relation'),
            ('CoordinateSet', 'coordinate_set'),
            ('SparseAlignment', 'alignment'),
            ('SparseQuantization', 'quantization'),
            ('PointVoxelMap', 'point_voxel_map'),
            ('CoordinateOrdering', 'coordinate_ordering'),
            ('SparseOccupancy', 'sparse_occupancy'),
            ('OccupancyExpansion', 'occupancy_expansion'),
            ('bytes', 'bytes'),
        )
        if marker in text
    }
    if len(candidates) == 1:
        return cast('IRValueType', candidates.pop())
    return 'any'


def function_output_type(function: Callable[..., Any]) -> IRValueType:
    """Infer an IR value type from a callable return annotation."""

    return annotation_value_type(
        inspect.signature(function).return_annotation
    )


def annotation_is_graph_value(annotation: object) -> bool:
    """Return whether an annotation represents a graph-carried value."""

    return annotation_value_type(annotation) != 'any'


def annotation_is_value_sequence(annotation: object) -> bool:
    """Return whether an annotation represents a sequence of graph values."""

    text = _annotation_text(annotation)
    return annotation_is_graph_value(annotation) and any(
        marker in text
        for marker in (
            'Sequence[',
            'collections.abc.Sequence[',
            'list[',
            'tuple[',
        )
    )


def _annotation_text(annotation: object) -> str:
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin is not None and args:
        return ' | '.join(_annotation_text(arg) for arg in args)
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, '__qualname__'):
        module = getattr(annotation, '__module__', '')
        qualname = getattr(annotation, '__qualname__', '')
        return f'{module}.{qualname}'
    return str(cast(object, annotation))
