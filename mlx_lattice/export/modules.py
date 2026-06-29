from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import mlx.core as mx
import mlx.nn as mxnn

from mlx_lattice.core import QuantizedWeight
from mlx_lattice.export._ops import field_value_type
from mlx_lattice.export.registry import (
    module_export_binding,
    operation_binding,
    validate_node_against_runtime,
)
from mlx_lattice.export.runtime import layout_id
from mlx_lattice.ir import (
    CURRENT_SCHEMA_VERSION,
    IRInputRef,
    IRManifest,
    IRNode,
    IRTensorSpec,
    IRValueType,
    ir_value_type,
)


@dataclass(frozen=True, slots=True)
class ExportedLatticeModel:
    """Manifest and weight tensors produced from a serializable module."""

    manifest: IRManifest
    weights: dict[str, mx.array]


@dataclass(frozen=True, slots=True)
class GraphOutput:
    """Public graph output mapping for explicit lattice graph export."""

    value: str
    value_type: IRValueType | None = None
    name: str | None = None


@runtime_checkable
class LatticeExportable(Protocol):
    """Protocol for modules that provide their own lattice export graph."""

    def export_lattice(
        self, builder: LatticeGraphBuilder, input_name: str
    ) -> str:
        """Append nodes to ``builder`` and return the output value name."""


class LatticeGraphBuilder:
    """Builder for explicit lattice model manifests.

    The builder is intentionally small: it owns graph nodes and weight tensors,
    validates op/module bindings through the shared runtime registry, and
    returns named values that can be wired into later nodes. It is the escape
    hatch for custom modules and DAGs without requiring Python tracing.
    """

    def __init__(
        self,
        input_name: str = 'input',
        *,
        inputs: Mapping[str, IRValueType] | None = None,
    ) -> None:
        self.input_name = input_name
        input_specs = (
            {input_name: 'sparse_tensor'} if inputs is None else inputs
        )
        self.inputs = {
            name: _value_type(value_type)
            for name, value_type in input_specs.items()
        }
        self.value_types: dict[str, IRValueType] = dict(self.inputs)
        self.nodes: list[IRNode] = []
        self.weights: dict[str, mx.array] = {}
        self._used_node_ids: set[str] = set()

    def add_op(
        self,
        name: str,
        op: str,
        *,
        inputs: Mapping[str, IRInputRef],
        output: str | None = None,
        outputs: Mapping[str, str] | None = None,
        parameters: Mapping[str, str] | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> str:
        """Append an operation node and return its primary output value."""

        binding = operation_binding(op)
        node_id = self.unique_name(name)
        primary = output or f'{node_id}.output'
        node_outputs = dict(outputs or {binding.output: primary})
        node = IRNode(
            id=node_id,
            op=op,
            inputs=dict(inputs),
            outputs=node_outputs,
            parameters=dict(parameters or {}),
            attributes=dict(attributes or {}),
        )
        validate_node_against_runtime(node)
        _validate_builder_node_types(
            node,
            binding.spec.input_types,
            binding.spec.value_attribute_types,
            self.value_types,
        )
        self.nodes.append(node)
        for port, value in node_outputs.items():
            self.value_types[value] = _infer_operation_output_type(op, port)
        return node_outputs.get(binding.output, primary)

    def call(
        self,
        op: str,
        /,
        name: str | None = None,
        *,
        output: str | None = None,
        outputs: Mapping[str, str] | None = None,
        parameters: Mapping[str, str] | None = None,
        **arguments: Any,
    ) -> str:
        """Append an operation by using its registered public argument names.

        Graph-value arguments are passed as value names. JSON-compatible
        constants become manifest attributes. Tensor parameters are passed
        either as artifact key strings or as arrays/packed ``QuantizedWeight``
        objects, which are stored automatically in ``weights.safetensors``.
        The ``parameters`` mapping remains available when the caller already
        owns stable artifact keys. This is the concise explicit-graph API;
        :meth:`add_op` remains the lower-level escape hatch when a manifest
        needs exact port dictionaries.
        """

        binding = operation_binding(op)
        node_name = name or _call_name(op)
        if parameters is None:
            node_parameters: dict[str, str] = {}
        else:
            node_parameters = dict(parameters)
        inputs: dict[str, IRInputRef] = {}
        attributes: dict[str, object] = {}
        remaining = set(arguments)
        for port in binding.input_arguments:
            if port in arguments:
                inputs[port] = _input_ref(arguments[port], port)
                remaining.remove(port)
        for name_ in binding.value_attribute_arguments:
            if name_ in arguments:
                value = arguments[name_]
                if not isinstance(value, str):
                    raise ValueError(
                        f'{op}.{name_} must be a graph value name.'
                    )
                attributes[name_] = value
                remaining.remove(name_)
        for name_ in binding.attribute_arguments:
            if name_ in arguments:
                attributes[name_] = arguments[name_]
                remaining.remove(name_)
        for name_ in binding.parameter_arguments:
            if name_ in arguments:
                value = arguments[name_]
                node_parameters[name_] = self._parameter_ref(
                    f'{node_name}.{name_}',
                    value,
                    binding.parameter_arguments[name_].kind,
                    op,
                    name_,
                )
                remaining.remove(name_)
        if remaining:
            raise ValueError(
                f'{op} received unsupported arguments: {sorted(remaining)}.'
            )
        return self.add_op(
            node_name,
            op,
            inputs=inputs,
            output=output,
            outputs=outputs,
            parameters=node_parameters,
            attributes=attributes,
        )

    def _parameter_ref(
        self,
        name: str,
        value: object,
        kind: str,
        op: str,
        argument: str,
    ) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, QuantizedWeight):
            if kind not in (
                'quantized_weight',
                'array_or_quantized_weight',
            ):
                raise ValueError(
                    f'{op}.{argument} does not accept QuantizedWeight.'
                )
            return self.add_quantized_parameter(name, value)
        if isinstance(value, mx.array):
            if kind == 'quantized_weight':
                raise ValueError(
                    f'{op}.{argument} requires QuantizedWeight or a packed '
                    'parameter key.'
                )
            return self.add_parameter(name, value)
        raise ValueError(
            f'{op}.{argument} must be a parameter key string, mx.array, '
            'or QuantizedWeight.'
        )

    def add_module(
        self,
        name: str,
        module: mxnn.Module,
        input_value: str,
        *,
        output: str | None = None,
    ) -> str:
        """Append a registered sparse NN module node."""

        binding = module_export_binding(module)
        node_id = self.unique_name(name)
        primary = output or f'{node_id}.output'
        parameters = self.module_parameters(
            node_id,
            module,
            binding.parameter_names,
        )
        node = IRNode(
            id=node_id,
            op=binding.op,
            inputs={'input': input_value},
            outputs={'output': primary},
            parameters=parameters,
            attributes=binding.attributes(module),
        )
        validate_node_against_runtime(node)
        op_binding = operation_binding(binding.op)
        _validate_builder_node_types(
            node,
            op_binding.spec.input_types,
            op_binding.spec.value_attribute_types,
            self.value_types,
        )
        self.nodes.append(node)
        self.value_types[primary] = _infer_operation_output_type(binding.op)
        return primary

    def output(
        self,
        value: str,
        *,
        name: str | None = None,
        value_type: IRValueType | None = None,
    ) -> GraphOutput:
        """Describe a public graph output, inferring its type by default."""

        return GraphOutput(
            value,
            self._output_type(value) if value_type is None else value_type,
            name,
        )

    def field(
        self,
        value: str,
        field: str,
        *,
        name: str | None = None,
    ) -> str:
        """Project a supported structural field from a graph value."""

        output = self.add_op(
            name or f'{value}.{field}',
            'value.field',
            inputs={'input': value},
            attributes={'field': field},
        )
        self.value_types[output] = field_value_type(
            self._output_type(value),
            field,
        )
        return output

    def add_parameter(self, name: str, value: mx.array) -> str:
        """Store a dense tensor parameter and return its artifact key."""

        key = self.unique_weight_name(name)
        self.weights[key] = value
        return key

    def add_quantized_parameter(
        self,
        name: str,
        value: QuantizedWeight,
    ) -> str:
        """Store a packed quantized parameter and return its artifact prefix."""

        prefix = self.unique_weight_name(name)
        self.weights[f'{prefix}.weight'] = value.weight
        self.weights[f'{prefix}.scales'] = value.scales
        self.weights[f'{prefix}.biases'] = value.biases
        self.weights[f'{prefix}.attrs'] = mx.array(
            [
                layout_id(value.layout),
                value.group_size,
                value.bits,
                value.in_channels,
                value.out_channels,
                *value.kernel_size,
            ],
            dtype=mx.int32,
        )
        return prefix

    def module_parameters(
        self,
        node_id: str,
        module: mxnn.Module,
        names: tuple[str, ...],
    ) -> dict[str, str]:
        """Export registered module parameters into artifact weights."""

        out: dict[str, str] = {}
        for name in names:
            if name == 'bias' and name not in module:
                continue
            if name == 'weight' and hasattr(module, '_quantized_weight'):
                out[name] = self.add_quantized_parameter(
                    f'{node_id}.{name}',
                    module._quantized_weight(),
                )
                continue
            if name == 'running_mean':
                if hasattr(module, 'running_mean'):
                    out['mean'] = self.add_parameter(
                        f'{node_id}.mean',
                        module.running_mean,
                    )
                continue
            if name == 'running_var':
                if hasattr(module, 'running_var'):
                    out['var'] = self.add_parameter(
                        f'{node_id}.var',
                        module.running_var,
                    )
                continue
            if name not in module:
                continue
            out[name] = self.add_parameter(
                f'{node_id}.{name}', module[name]
            )
        return out

    def unique_name(self, name: str) -> str:
        """Return a graph-unique sanitized node name."""

        base = _safe_name(name)
        candidate = base
        index = 1
        while candidate in self._used_node_ids:
            candidate = f'{base}_{index}'
            index += 1
        self._used_node_ids.add(candidate)
        return candidate

    def unique_weight_name(self, name: str) -> str:
        """Return an artifact-unique sanitized weight name."""

        base = _safe_name(name)
        candidate = base
        index = 1
        while (
            candidate in self.weights
            or f'{candidate}.weight' in self.weights
            or f'{candidate}.attrs' in self.weights
        ):
            candidate = f'{base}_{index}'
            index += 1
        return candidate

    def manifest(
        self,
        *,
        outputs: Mapping[str, IRValueType | GraphOutput | None]
        | Sequence[str]
        | None = None,
        output_name: str = 'output',
        output_value: str | None = None,
        input_type: IRValueType | None = None,
        output_type: IRValueType | None = None,
        producer: Mapping[str, str] | None = None,
    ) -> IRManifest:
        """Build an immutable manifest from the accumulated graph."""

        if input_type is not None:
            self.inputs[self.input_name] = input_type
        if outputs is None:
            renamed_nodes = _rename_final_output(
                self.nodes,
                self._require_output(output_value),
                output_name,
            )
            value_type = (
                self._output_type(output_value)
                if output_type is None
                else _value_type(output_type)
            )
            output_tensors = (IRTensorSpec(output_name, value_type),)
        else:
            output_specs = tuple(
                self._graph_output(value, spec)
                for value, spec in _output_items(outputs)
            )
            renamed_nodes = _rename_outputs(
                self.nodes,
                {
                    value: output_spec.name
                    for value, output_spec in zip(
                        outputs, output_specs, strict=True
                    )
                    if output_spec.name != value
                },
            )
            output_tensors = output_specs
        return IRManifest(
            schema_version=CURRENT_SCHEMA_VERSION,
            producer=dict(producer or {'name': 'mlx-lattice'}),
            runtime={'name': 'mlx-lattice', 'version': '>=0.2,<0.3'},
            inputs=tuple(
                IRTensorSpec(name, _value_type(value_type))
                for name, value_type in self.inputs.items()
            ),
            outputs=output_tensors,
            nodes=tuple(renamed_nodes),
        )

    def _require_output(self, value: str | None) -> str:
        if value is None:
            raise ValueError('output_value is required.')
        return value

    def _output_type(self, value: str | None) -> IRValueType:
        if value is None:
            raise ValueError('output_value is required.')
        try:
            return self.value_types[value]
        except KeyError as exc:
            raise ValueError(
                f'cannot infer graph output type for {value!r}; pass an '
                'explicit output type.'
            ) from exc

    def _graph_output(
        self,
        value: str,
        spec: IRValueType | GraphOutput | None,
    ) -> IRTensorSpec:
        if isinstance(spec, GraphOutput):
            if spec.value != value:
                raise ValueError(
                    f'graph output key {value!r} does not match output value '
                    f'{spec.value!r}.'
                )
            value_type = (
                self._output_type(value)
                if spec.value_type is None
                else _value_type(spec.value_type)
            )
            return IRTensorSpec(spec.name or value, value_type)
        if spec is None:
            return IRTensorSpec(value, self._output_type(value))
        return IRTensorSpec(value, _value_type(spec))


def export_lattice_graph(
    builder: LatticeGraphBuilder,
    *,
    outputs: Mapping[str, IRValueType | GraphOutput | None] | Sequence[str],
    producer: Mapping[str, str] | None = None,
) -> ExportedLatticeModel:
    """Export an explicitly built lattice graph."""

    return ExportedLatticeModel(
        manifest=builder.manifest(outputs=outputs, producer=producer),
        weights=builder.weights,
    )


def export_lattice_module(
    module: mxnn.Module,
    *,
    input_name: str = 'input',
    output_name: str = 'output',
    input_type: IRValueType = 'sparse_tensor',
    output_type: IRValueType | None = None,
    producer: Mapping[str, str] | None = None,
) -> ExportedLatticeModel:
    """Export a sparse NN module graph.

    Built-in lattice modules and sequential containers export structurally.
    Custom modules can implement ``export_lattice(builder, input_name)`` to
    emit arbitrary DAGs with the same builder used internally.
    """

    builder = LatticeGraphBuilder(
        input_name, inputs={input_name: input_type}
    )
    output_value = _export_module(module, builder, input_name)
    return ExportedLatticeModel(
        manifest=builder.manifest(
            output_name=output_name,
            output_value=output_value,
            input_type=input_type,
            output_type=output_type,
            producer=producer,
        ),
        weights=builder.weights,
    )


def _export_module(
    module: mxnn.Module,
    builder: LatticeGraphBuilder,
    input_value: str,
) -> str:
    if isinstance(module, LatticeExportable):
        return module.export_lattice(builder, input_value)

    children = _ordered_children(module)
    if not children:
        return builder.add_module('module', module, input_value)

    current = input_value
    for name, child in children:
        current = _export_child(name, child, builder, current)
    return current


def _export_child(
    name: str,
    child: object,
    builder: LatticeGraphBuilder,
    input_value: str,
) -> str:
    if isinstance(child, mxnn.Module):
        if isinstance(child, LatticeExportable):
            return child.export_lattice(builder, input_value)
        grandchildren = _ordered_children(child)
        if grandchildren:
            current = input_value
            for child_name, grandchild in grandchildren:
                current = _export_child(
                    f'{name}_{child_name}',
                    grandchild,
                    builder,
                    current,
                )
            return current
        return builder.add_module(name, child, input_value)

    if isinstance(child, list | tuple):
        current = input_value
        for index, item in enumerate(child):
            current = _export_child(
                f'{name}_{index}', item, builder, current
            )
        return current

    raise ValueError(f'child {name!r} is not an exportable MLX module.')


def _ordered_children(
    module: mxnn.Module,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(name), child) for name, child in module.children().items()
    )


def _rename_final_output(
    nodes: list[IRNode],
    current: str,
    output_name: str,
) -> list[IRNode]:
    return _rename_outputs(nodes, {current: output_name})


def _rename_outputs(
    nodes: list[IRNode],
    replacements: Mapping[str, str],
) -> list[IRNode]:
    if not nodes:
        raise ValueError('cannot export an empty lattice module graph.')
    if not replacements:
        return list(nodes)
    renamed: list[IRNode] = []
    for node in nodes:
        binding = operation_binding(node.op)
        inputs = {
            port: _replace_input_ref(value, replacements)
            for port, value in node.inputs.items()
        }
        outputs = {
            port: replacements.get(value, value)
            for port, value in node.outputs.items()
        }
        attributes = {
            name: replacements.get(value, value)
            if name in binding.value_attribute_arguments
            and isinstance(value, str)
            else value
            for name, value in node.attributes.items()
        }
        renamed.append(
            IRNode(
                id=node.id,
                op=node.op,
                inputs=inputs,
                outputs=outputs,
                parameters=node.parameters,
                attributes=attributes,
                support=node.support,
            )
        )
    return renamed


def _replace_input_ref(
    value: IRInputRef,
    replacements: Mapping[str, str],
) -> IRInputRef:
    if isinstance(value, str):
        return replacements.get(value, value)
    return tuple(replacements.get(item, item) for item in value)


def _validate_builder_node_types(
    node: IRNode,
    input_types: Mapping[str, IRValueType],
    value_attribute_types: Mapping[str, IRValueType],
    values: Mapping[str, IRValueType],
) -> None:
    for port, value_ref in node.inputs.items():
        expected_type = input_types.get(port, 'any')
        for value_name in _input_ref_names(value_ref):
            _validate_builder_value_type(
                node.id,
                f'inputs.{port}',
                value_name,
                expected_type,
                values.get(value_name, 'any'),
            )
    for name, expected_type in value_attribute_types.items():
        if name not in node.attributes:
            continue
        value_name = node.attributes[name]
        if not isinstance(value_name, str):
            continue
        _validate_builder_value_type(
            node.id,
            f'attributes.{name}',
            value_name,
            expected_type,
            values.get(value_name, 'any'),
        )


def _validate_builder_value_type(
    node_id: str,
    path: str,
    value_name: str,
    expected_type: IRValueType,
    actual: IRValueType,
) -> None:
    if expected_type == 'any' or actual == 'any' or expected_type == actual:
        return
    raise ValueError(
        f'{node_id}.{path} expects {expected_type!r} but graph value '
        f'{value_name!r} has type {actual!r}.'
    )


def _input_ref_names(value: IRInputRef) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _input_ref(value: object, name: str) -> IRInputRef:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and all(
        isinstance(item, str) for item in value
    ):
        return cast('tuple[str, ...]', tuple(value))
    raise ValueError(
        f'{name} must be a graph value name or sequence of names.'
    )


def _call_name(op: str) -> str:
    return op.removeprefix('ops.').replace('.', '_')


def _value_type(value: str) -> IRValueType:
    return ir_value_type(value)


def _output_items(
    outputs: Mapping[str, IRValueType | GraphOutput | None] | Sequence[str],
) -> tuple[tuple[str, IRValueType | GraphOutput | None], ...]:
    if isinstance(outputs, Mapping):
        return cast(
            'tuple[tuple[str, IRValueType | GraphOutput | None], ...]',
            tuple(outputs.items()),
        )
    return tuple((value, None) for value in outputs)


def _infer_operation_output_type(
    op: str,
    port: str = 'output',
) -> IRValueType:
    binding = operation_binding(op)
    return binding.spec.output_types.get(port, 'any')


def _safe_name(name: str) -> str:
    clean = ''.join(
        char if char.isalnum() or char == '_' else '_' for char in name
    )
    return clean.strip('_') or 'module'
