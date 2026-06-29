from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import mlx.core as mx
import mlx.nn as mxnn
from lattice_contract import DTypePolicy, IRNode, IROpSpec, IRValueType
from lattice_contract.manifest import IRInputRef

from mlx_lattice.core import QuantizedWeight, SparseTensor
from mlx_lattice.core.coords import (
    CoordinateOrdering,
    CoordinateSet,
    OccupancyExpansion,
    PointVoxelMap,
    SparseAlignment,
    SparseOccupancy,
    SparseQuantization,
)
from mlx_lattice.core.relations import KernelRelation, NeighborRelation
from mlx_lattice.core.types import Triple

type GraphValue = (
    SparseTensor
    | mx.array
    | KernelRelation
    | NeighborRelation
    | CoordinateSet
    | SparseAlignment
    | SparseQuantization
    | PointVoxelMap
    | CoordinateOrdering
    | SparseOccupancy
    | OccupancyExpansion
    | bytes
    | object
)
type GraphHandler = Callable[
    ['ExecutionContext', IRNode], dict[str, GraphValue]
]
ParamKind = Literal[
    'array',
    'optional_array',
    'quantized_weight',
    'array_or_quantized_weight',
]


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    """How an IR parameter name is converted into a runtime argument."""

    argument: str
    kind: ParamKind = 'array'


@dataclass(frozen=True, slots=True)
class ValueTypeBinding:
    """Runtime type and public fields for one IR value type."""

    value_type: IRValueType
    runtime_type: type | tuple[type, ...]
    fields: Mapping[str, IRValueType] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OperationBinding:
    """Runtime binding for one public lattice operation."""

    spec: IROpSpec
    function: Callable[..., GraphValue]
    input_arguments: Mapping[str, str]
    parameter_arguments: Mapping[str, ParameterBinding] = field(
        default_factory=dict
    )
    attribute_arguments: Mapping[str, str] = field(default_factory=dict)
    value_attribute_arguments: Mapping[str, str] = field(
        default_factory=dict
    )
    defaults: Mapping[str, Any] = field(default_factory=dict)
    sequence_inputs: frozenset[str] = frozenset()
    output: str = 'output'
    handler: GraphHandler | None = None

    def run(
        self,
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        if self.handler is not None:
            return self.handler(context, node)
        return self.run_default(context, node)

    def run_default(
        self,
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        kwargs = self.arguments(context, node)
        return {self.output: self.function(**kwargs)}

    def arguments(
        self,
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, Any]:
        kwargs = dict(self.defaults)
        kwargs.update(
            {
                argument: context.input_value(
                    node.inputs[port],
                    sequence=port in self.sequence_inputs,
                )
                for port, argument in self.input_arguments.items()
            }
        )
        kwargs.update(
            {
                binding.argument: context.parameter(
                    node.parameters[name],
                    binding.kind,
                )
                for name, binding in self.parameter_arguments.items()
                if name in node.parameters
            }
        )
        kwargs.update(
            {
                argument: node.attributes[name]
                for name, argument in self.attribute_arguments.items()
                if name in node.attributes
            }
        )
        kwargs.update(
            {
                argument: context.value(node.attributes[name])
                for name, argument in self.value_attribute_arguments.items()
                if name in node.attributes
            }
        )
        return kwargs


@dataclass(frozen=True, slots=True)
class ModuleBinding:
    """Artifact binding for a serializable sparse NN module."""

    module_type: type[mxnn.Module]
    op: str
    parameter_names: tuple[str, ...] = ()
    attributes: Callable[[mxnn.Module], dict[str, Any]] = lambda _: {}


@dataclass(slots=True)
class ExecutionContext:
    """Mutable graph execution state."""

    values: dict[str, GraphValue]
    weights: Mapping[str, mx.array]
    batch_size: int | None = None

    def value(self, name: str) -> GraphValue:
        try:
            return self.values[name]
        except KeyError as exc:
            raise ValueError(f'missing graph value {name!r}.') from exc

    def input_value(
        self,
        value_ref: IRInputRef,
        *,
        sequence: bool = False,
    ) -> GraphValue | tuple[GraphValue, ...]:
        if isinstance(value_ref, str):
            value = self.value(value_ref)
            return (value,) if sequence else value
        values = tuple(self.value(name) for name in value_ref)
        return values if sequence else _single_value(values)

    def sparse(self, name: str) -> SparseTensor:
        value = self.value(name)
        if not isinstance(value, SparseTensor):
            raise ValueError(f'graph value {name!r} is not a SparseTensor.')
        return value

    def parameter(self, name: str, kind: ParamKind = 'array') -> Any:
        if kind == 'optional_array':
            return None if name == '' else self.array(name)
        if kind == 'quantized_weight':
            return self.quantized_weight(name)
        if kind == 'array_or_quantized_weight':
            return (
                self.quantized_weight(name)
                if f'{name}.attrs' in self.weights
                else self.array(name)
            )
        return self.array(name)

    def array(self, name: str) -> mx.array:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise ValueError(f'missing lattice weight {name!r}.') from exc

    def optional_array(self, name: str | None) -> mx.array | None:
        return None if name is None else self.array(name)

    def quantized_weight(self, prefix: str) -> QuantizedWeight:
        attrs = self.weights[f'{prefix}.attrs']
        raw_value = attrs.tolist()
        if not isinstance(raw_value, list):
            raise ValueError(
                f'quantized parameter {prefix!r} metadata must be a vector.'
            )
        raw = [int(item) for item in raw_value]
        if len(raw) != 8:
            raise ValueError(
                f'quantized parameter {prefix!r} has invalid metadata.'
            )
        (
            layout_id_value,
            group_size,
            bits,
            in_channels,
            out_channels,
            *kernel,
        ) = raw
        return QuantizedWeight(
            self.array(f'{prefix}.weight'),
            self.array(f'{prefix}.scales'),
            self.array(f'{prefix}.biases'),
            group_size,
            bits,
            in_channels,
            out_channels,
            cast('Triple', tuple(kernel)),
            _layout_from_id(layout_id_value),
        )


def infer_batch_size(values: Mapping[str, GraphValue]) -> int | None:
    """Infer batch size from sparse graph inputs when metadata exists."""

    for value in values.values():
        if (
            isinstance(value, SparseTensor)
            and value.batch_counts is not None
        ):
            return len(value.batch_counts)
    return None


def apply_dtype_policy(
    value: GraphValue,
    policy: DTypePolicy,
    *,
    inference: bool = True,
) -> GraphValue:
    """Apply an artifact dtype policy to graph-carried tensor values."""

    if policy == 'preserve':
        return value
    if policy == 'fp16_inference' and not inference:
        return value
    target = _dtype_policy_target(policy)
    if isinstance(value, SparseTensor):
        return value.replace(feats=_cast_float_array(value.feats, target))
    if isinstance(value, mx.array):
        return _cast_float_array(value, target)
    return value


def validate_value_type(
    name: str,
    expected: IRValueType,
    value: GraphValue,
) -> None:
    """Validate a runtime value against an IR value type."""

    binding = value_type_binding(expected)
    if binding is None:
        raise ValueError(f'unsupported IR value type: {expected!r}.')
    if binding.runtime_type is object:
        return
    if not isinstance(value, binding.runtime_type):
        raise ValueError(
            f'graph value {name!r} must be '
            f'{_value_type_label(binding.runtime_type)}.'
        )


def layout_id(value: str) -> int:
    """Encode a quantized weight layout for artifact metadata."""

    if value == 'linear':
        return 0
    if value == 'kernel_major':
        return 1
    if value == 'dense_5d':
        return 2
    raise ValueError(f'unknown quantized weight layout: {value!r}.')


def value_type_binding(value_type: IRValueType) -> ValueTypeBinding | None:
    """Return the artifact binding for an IR value type."""

    return _VALUE_TYPES.get(value_type)


def value_type_fields(value_type: IRValueType) -> Mapping[str, IRValueType]:
    """Return public structural fields exposed for an IR value type."""

    binding = value_type_binding(value_type)
    return {} if binding is None else binding.fields


def iter_value_type_bindings() -> tuple[ValueTypeBinding, ...]:
    """Return artifact bindings for all known IR value types."""

    return tuple(_VALUE_TYPES.values())


def _single_value(values: tuple[GraphValue, ...]) -> GraphValue:
    if len(values) != 1:
        raise ValueError('expected a single graph value reference.')
    return values[0]


def _dtype_policy_target(policy: DTypePolicy):
    if policy == 'fp32':
        return mx.float32
    if policy in ('fp16', 'fp16_inference'):
        return mx.float16
    raise ValueError(f'unsupported dtype policy: {policy!r}.')


def _cast_float_array(value: mx.array, dtype) -> mx.array:
    if value.dtype in (mx.float16, mx.float32):
        return value.astype(dtype)
    return value


def _layout_from_id(value: int):
    if value == 0:
        return 'linear'
    if value == 1:
        return 'kernel_major'
    if value == 2:
        return 'dense_5d'
    raise ValueError(f'unknown quantized weight layout id: {value}.')


def _value_type_label(runtime_type: type | tuple[type, ...]) -> str:
    if isinstance(runtime_type, tuple):
        return ' or '.join(_value_type_label(item) for item in runtime_type)
    if runtime_type is mx.array:
        return 'a dense tensor'
    return runtime_type.__name__


def _fields(**fields: IRValueType) -> Mapping[str, IRValueType]:
    return fields


_VALUE_TYPES: dict[IRValueType, ValueTypeBinding] = {
    'any': ValueTypeBinding('any', object),
    'sparse_tensor': ValueTypeBinding('sparse_tensor', SparseTensor),
    'dense_tensor': ValueTypeBinding('dense_tensor', mx.array),
    'relation': ValueTypeBinding(
        'relation',
        (KernelRelation, NeighborRelation),
    ),
    'coordinate_set': ValueTypeBinding(
        'coordinate_set',
        CoordinateSet,
        _fields(coords='dense_tensor', active_rows='dense_tensor'),
    ),
    'alignment': ValueTypeBinding(
        'alignment',
        SparseAlignment,
        _fields(
            coords='dense_tensor',
            active_rows='dense_tensor',
            lhs_rows='dense_tensor',
            rhs_rows='dense_tensor',
        ),
    ),
    'quantization': ValueTypeBinding(
        'quantization',
        SparseQuantization,
        _fields(
            coords='dense_tensor',
            active_rows='dense_tensor',
            inverse_rows='dense_tensor',
            counts='dense_tensor',
        ),
    ),
    'point_voxel_map': ValueTypeBinding(
        'point_voxel_map',
        PointVoxelMap,
        _fields(rows='dense_tensor', weights='dense_tensor'),
    ),
    'coordinate_ordering': ValueTypeBinding(
        'coordinate_ordering',
        CoordinateOrdering,
        _fields(
            coords='dense_tensor',
            order='dense_tensor',
            inverse_rows='dense_tensor',
        ),
    ),
    'sparse_occupancy': ValueTypeBinding(
        'sparse_occupancy',
        SparseOccupancy,
        _fields(
            coords='dense_tensor',
            active_rows='dense_tensor',
            occupancy='dense_tensor',
        ),
    ),
    'occupancy_expansion': ValueTypeBinding(
        'occupancy_expansion',
        OccupancyExpansion,
        _fields(
            coords='dense_tensor',
            active_rows='dense_tensor',
            parent_rows='dense_tensor',
            child_indices='dense_tensor',
        ),
    ),
    'bytes': ValueTypeBinding('bytes', bytes),
}
