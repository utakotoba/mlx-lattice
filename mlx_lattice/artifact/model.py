from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import mlx.core as mx
from lattice_contract import (
    IRInputRef,
    IRManifest,
    IRTensorSpec,
    IRValueType,
)

from mlx_lattice._native import backend_info
from mlx_lattice.artifact.bindings import (
    ExecutionContext,
    GraphValue,
    ParameterBinding,
    apply_dtype_policy,
    infer_batch_size,
    validate_value_type,
)
from mlx_lattice.artifact.registry import (
    operation_binding,
    validate_node_against_artifact,
)


class LatticeModel:
    """Executable in-memory graph loaded from a lattice artifact.

    The model does not execute generated Python. It validates IR nodes and
    dispatches each node through the public ``mlx_lattice.ops`` surface.
    """

    def __init__(
        self,
        manifest: IRManifest,
        weights: Mapping[str, mx.array],
    ) -> None:
        self.manifest = manifest
        self.weights = _policy_weights(manifest, weights)
        _validate_runtime_compatibility(manifest)
        _validate_manifest(manifest, self.weights)

    def __call__(self, *inputs: GraphValue, **named: GraphValue):
        values = _bind_inputs(
            self.manifest,
            self.manifest.inputs,
            inputs,
            named,
        )
        context = ExecutionContext(
            values,
            self.weights,
            batch_size=infer_batch_size(values),
        )
        for node in self.manifest.nodes:
            binding = operation_binding(node.op)
            outputs = binding.run(context, node)
            _validate_node_outputs(node, binding.spec.output_types, outputs)
            for port, value in outputs.items():
                context.values[node.outputs[port]] = _policy_value(
                    self.manifest,
                    value,
                )
        return _collect_outputs(self.manifest.outputs, context)


def _validate_node_outputs(
    node,
    output_types: Mapping[str, IRValueType],
    outputs: Mapping[str, GraphValue],
) -> None:
    actual_ports = set(outputs)
    expected_ports = set(node.outputs)
    missing = expected_ports - actual_ports
    extra = actual_ports - expected_ports
    if missing:
        raise ValueError(
            f'{node.id}.artifact_outputs missing required keys: '
            f'{sorted(missing)}.'
        )
    if extra:
        raise ValueError(
            f'{node.id}.artifact_outputs has unsupported keys: '
            f'{sorted(extra)}.'
        )
    for port, value in outputs.items():
        validate_value_type(
            f'{node.id}.outputs.{port}',
            output_types.get(port, 'any'),
            value,
        )


def _validate_manifest(
    manifest: IRManifest,
    weights: Mapping[str, mx.array],
) -> None:
    values = {item.name: item.type for item in manifest.inputs}
    for node in manifest.nodes:
        validate_node_against_artifact(node)
        binding = operation_binding(node.op)
        _validate_node_input_types(node, binding.spec.input_types, values)
        for name in binding.value_attribute_arguments:
            if name in node.attributes:
                ref = node.attributes[name]
                if ref not in values:
                    raise ValueError(
                        f'{node.id}.attributes.{name} references unknown '
                        f'graph value {ref!r}.'
                    )
                _validate_graph_value_type(
                    node.id,
                    f'attributes.{name}',
                    str(ref),
                    binding.spec.value_attribute_types.get(name, 'any'),
                    values[str(ref)],
                )
        for name, parameter in binding.parameter_arguments.items():
            if name in node.parameters:
                _validate_weight(
                    node.id,
                    node.parameters[name],
                    parameter,
                    weights,
                )
        for port, value in node.outputs.items():
            values[value] = binding.spec.output_types.get(port, 'any')
    for spec in manifest.outputs:
        if spec.name not in values:
            raise ValueError(f'unknown manifest output {spec.name!r}.')
        _validate_graph_value_type(
            'manifest',
            f'outputs.{spec.name}',
            spec.name,
            spec.type,
            values[spec.name],
        )


def _validate_runtime_compatibility(manifest: IRManifest) -> None:
    name = manifest.runtime.get('name')
    if name not in (None, '', 'mlx-lattice'):
        raise ValueError(
            f"manifest runtime.name must be 'mlx-lattice', got {name!r}."
        )
    specifier = manifest.runtime.get('version')
    if specifier in (None, ''):
        return
    version = str(backend_info()['version'])
    if not _version_satisfies(version, specifier):
        raise ValueError(
            f'manifest runtime.version {specifier!r} is not compatible '
            f'with mlx-lattice runtime {version!r}.'
        )


def _version_satisfies(version: str, specifier: str) -> bool:
    version_tuple = _version_tuple(version)
    for clause in (item.strip() for item in specifier.split(',')):
        if not clause:
            continue
        op, expected = _version_clause(clause)
        expected_tuple = _version_tuple(expected)
        comparison = _compare_version_tuples(version_tuple, expected_tuple)
        if op == '==' and comparison != 0:
            return False
        if op == '!=' and comparison == 0:
            return False
        if op == '>=' and comparison < 0:
            return False
        if op == '>' and comparison <= 0:
            return False
        if op == '<=' and comparison > 0:
            return False
        if op == '<' and comparison >= 0:
            return False
    return True


def _version_clause(clause: str) -> tuple[str, str]:
    for op in ('>=', '<=', '==', '!=', '>', '<'):
        if clause.startswith(op):
            value = clause.removeprefix(op).strip()
            if not value:
                break
            return op, value
    if clause and clause[0].isdigit():
        return '==', clause
    raise ValueError(f'unsupported runtime version specifier: {clause!r}.')


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for part in value.split('.'):
        digits = ''.join(char for char in _leading_digits(part))
        if digits == '':
            break
        parts.append(int(digits))
    if not parts:
        raise ValueError(f'unsupported runtime version value: {value!r}.')
    return tuple(parts)


def _leading_digits(value: str) -> str:
    digits = []
    for char in value:
        if not char.isdigit():
            break
        digits.append(char)
    return ''.join(digits)


def _compare_version_tuples(
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
) -> int:
    size = max(len(lhs), len(rhs))
    left = lhs + (0,) * (size - len(lhs))
    right = rhs + (0,) * (size - len(rhs))
    return (left > right) - (left < right)


def _validate_node_input_types(
    node,
    expected: Mapping[str, IRValueType],
    values: Mapping[str, IRValueType],
) -> None:
    for port, value_ref in node.inputs.items():
        expected_type = expected.get(port, 'any')
        for value_name in _value_refs(value_ref):
            if value_name not in values:
                continue
            _validate_graph_value_type(
                node.id,
                f'inputs.{port}',
                value_name,
                expected_type,
                values[value_name],
            )


def _validate_graph_value_type(
    node_id: str,
    path: str,
    value_name: str,
    expected: IRValueType,
    actual: IRValueType,
) -> None:
    if expected == 'any' or actual == 'any' or expected == actual:
        return
    raise ValueError(
        f'{node_id}.{path} expects {expected!r} but graph value '
        f'{value_name!r} has type {actual!r}.'
    )


def _value_refs(value: IRInputRef) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _validate_weight(
    node_id: str,
    name: str,
    binding: ParameterBinding,
    weights: Mapping[str, mx.array],
) -> None:
    if binding.kind in ('quantized_weight', 'array_or_quantized_weight'):
        if _has_quantized_weight(name, weights):
            return
        if binding.kind == 'quantized_weight':
            _raise_missing_quantized_weight(node_id, name, weights)
        if name in weights:
            return
    if name not in weights:
        raise ValueError(f'{node_id}.parameters missing weight {name!r}.')


def _has_quantized_weight(
    name: str,
    weights: Mapping[str, mx.array],
) -> bool:
    return all(
        f'{name}.{suffix}' in weights
        for suffix in ('weight', 'scales', 'biases', 'attrs')
    )


def _raise_missing_quantized_weight(
    node_id: str,
    name: str,
    weights: Mapping[str, mx.array],
) -> None:
    missing = [
        key
        for key in (
            f'{name}.weight',
            f'{name}.scales',
            f'{name}.biases',
            f'{name}.attrs',
        )
        if key not in weights
    ]
    raise ValueError(
        f'{node_id}.parameters has missing quantized weight tensors: '
        f'{missing}.'
    )


def _bind_inputs(
    manifest: IRManifest,
    specs: tuple[IRTensorSpec, ...],
    inputs: tuple[GraphValue, ...],
    named: dict[str, GraphValue],
) -> dict[str, GraphValue]:
    if len(inputs) > len(specs):
        raise ValueError('too many positional graph inputs.')
    values: dict[str, GraphValue] = {}
    for spec, value in zip(specs, inputs, strict=False):
        _validate_value_type(spec.name, spec.type, value)
        values[spec.name] = _policy_value(manifest, value)
    for spec in specs[len(inputs) :]:
        if spec.name not in named:
            raise ValueError(f'missing graph input {spec.name!r}.')
        value = named.pop(spec.name)
        _validate_value_type(spec.name, spec.type, value)
        values[spec.name] = _policy_value(manifest, value)
    if named:
        raise ValueError(f'unknown graph inputs: {sorted(named)}.')
    return values


def _validate_value_type(
    name: str,
    expected: IRValueType,
    value: GraphValue,
) -> None:
    validate_value_type(name, expected, value)


def _collect_outputs(
    specs: tuple[IRTensorSpec, ...], context: ExecutionContext
):
    outputs = []
    for spec in specs:
        value = context.values[spec.name]
        _validate_value_type(spec.name, spec.type, value)
        outputs.append(value)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _policy_weights(
    manifest: IRManifest,
    weights: Mapping[str, mx.array],
) -> dict[str, mx.array]:
    quantized_payloads = _quantized_payload_names(weights)
    return {
        name: cast(
            'mx.array',
            value
            if name in quantized_payloads
            else apply_dtype_policy(value, manifest.dtype_policy),
        )
        for name, value in weights.items()
    }


def _policy_value(manifest: IRManifest, value: GraphValue) -> GraphValue:
    return apply_dtype_policy(value, manifest.dtype_policy)


def _quantized_payload_names(weights: Mapping[str, mx.array]) -> set[str]:
    prefixes = {
        name.removesuffix('.attrs')
        for name in weights
        if name.endswith('.attrs')
        and f'{name.removesuffix(".attrs")}.weight' in weights
        and f'{name.removesuffix(".attrs")}.scales' in weights
        and f'{name.removesuffix(".attrs")}.biases' in weights
    }
    return {
        f'{prefix}.{suffix}'
        for prefix in prefixes
        for suffix in ('weight', 'scales', 'biases', 'attrs')
    }
