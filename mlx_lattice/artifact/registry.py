from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

import mlx.nn as mxnn
from lattice_contract import IRNode, IROpSpec, IRValueType

import mlx_lattice.nn as lnn
from mlx_lattice.artifact.bindings import (
    GraphHandler,
    GraphValue,
    ModuleBinding,
    OperationBinding,
    ParameterBinding,
)
from mlx_lattice.artifact.ops import register_operations
from mlx_lattice.nn._artifact import (
    ModuleArtifactSpec,
    annotation_value_type,
    function_output_type,
    module_artifact_spec,
)

HandlerT = TypeVar('HandlerT', bound=GraphHandler)
ModuleT = TypeVar('ModuleT', bound=mxnn.Module)

_OPS: dict[str, OperationBinding] = {}
_MODULES: dict[type[mxnn.Module], ModuleBinding] = {}


class _OperationRegistrar:
    """Small callable adapter used by operation registration modules."""

    def __call__(
        self,
        name: str,
        *,
        function: Callable[..., GraphValue],
        inputs: Mapping[str, str],
        outputs: set[str] | None = None,
        parameters: Mapping[str, str | ParameterBinding] | None = None,
        optional_parameters: Mapping[str, str | ParameterBinding]
        | None = None,
        attributes: Mapping[str, str] | None = None,
        value_attributes: Mapping[str, str] | None = None,
        defaults: Mapping[str, Any] | None = None,
        sequence_inputs: set[str] | None = None,
        output: str = 'output',
        output_types: Mapping[str, IRValueType] | None = None,
        input_types: Mapping[str, IRValueType] | None = None,
        value_attribute_types: Mapping[str, IRValueType] | None = None,
        handler: GraphHandler | None = None,
    ) -> Callable[[HandlerT], HandlerT]:
        return lattice_op(
            name,
            function=function,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            optional_parameters=optional_parameters,
            attributes=attributes,
            value_attributes=value_attributes,
            defaults=defaults,
            sequence_inputs=sequence_inputs,
            output=output,
            output_types=output_types,
            input_types=input_types,
            value_attribute_types=value_attribute_types,
            handler=handler,
        )

    def binding(self, name: str) -> OperationBinding:
        """Return the registered binding for ``name``."""

        return operation_binding(name)


def lattice_op(
    name: str,
    *,
    function: Callable[..., GraphValue],
    inputs: Mapping[str, str],
    outputs: set[str] | None = None,
    parameters: Mapping[str, str | ParameterBinding] | None = None,
    optional_parameters: Mapping[str, str | ParameterBinding] | None = None,
    attributes: Mapping[str, str] | None = None,
    value_attributes: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
    sequence_inputs: set[str] | None = None,
    output: str = 'output',
    output_types: Mapping[str, IRValueType] | None = None,
    input_types: Mapping[str, IRValueType] | None = None,
    value_attribute_types: Mapping[str, IRValueType] | None = None,
    handler: GraphHandler | None = None,
) -> Callable[[HandlerT], HandlerT]:
    """Register a public operation with one compact annotation."""

    parameter_bindings = _parameter_bindings(parameters or {})
    optional_bindings = _parameter_bindings(optional_parameters or {})
    spec = IROpSpec(
        name=name,
        inputs=frozenset(inputs),
        outputs=frozenset(outputs or {output}),
        output_types=dict(
            output_types or {output: function_output_type(function)}
        ),
        input_types=dict(input_types or _argument_types(function, inputs)),
        value_attribute_types=dict(
            value_attribute_types
            or _argument_types(function, value_attributes or {})
        ),
        parameters=frozenset(parameter_bindings),
        optional_parameters=frozenset(optional_bindings),
        attributes=frozenset(attributes or ()),
        value_attributes=frozenset(value_attributes or ()),
    )

    def decorator(default_handler: HandlerT) -> HandlerT:
        if name in _OPS:
            raise ValueError(
                f'duplicate lattice operation binding: {name}.'
            )
        _OPS[name] = OperationBinding(
            spec=spec,
            function=function,
            input_arguments=dict(inputs),
            parameter_arguments={**parameter_bindings, **optional_bindings},
            attribute_arguments=dict(attributes or {}),
            value_attribute_arguments=dict(value_attributes or {}),
            defaults=dict(defaults or {}),
            sequence_inputs=frozenset(sequence_inputs or ()),
            output=output,
            handler=handler or default_handler,
        )
        return default_handler

    return decorator


def module_binding[ModuleT: mxnn.Module](
    module_type: type[ModuleT],
    *,
    op: str,
    parameter_names: Sequence[str] = (),
    attributes: Callable[[ModuleT], dict[str, Any]] = lambda _: {},
) -> Callable[[type[ModuleT]], type[ModuleT]]:
    """Register a serializable NN module producer."""

    def decorator(cls: type[ModuleT]) -> type[ModuleT]:
        if module_type in _MODULES:
            raise ValueError(
                f'duplicate lattice module binding: {module_type!r}.'
            )
        _MODULES[module_type] = ModuleBinding(
            module_type=module_type,
            op=op,
            parameter_names=tuple(parameter_names),
            attributes=lambda module: attributes(cast(ModuleT, module)),
        )
        return cls

    return decorator


def iter_operation_specs() -> tuple[IROpSpec, ...]:
    """Return all operation contracts supported by the MLX artifact graph."""

    return tuple(binding.spec for binding in _OPS.values())


def operation_spec(name: str) -> IROpSpec:
    """Return the artifact operation spec for ``name``."""

    return operation_binding(name).spec


def operation_binding(name: str) -> OperationBinding:
    """Return the artifact operation binding for ``name``."""

    try:
        return _OPS[name]
    except KeyError as exc:
        raise ValueError(
            f'unsupported lattice IR op for MLX artifact: {name!r}.'
        ) from exc


def module_artifact_binding(module: mxnn.Module) -> ModuleBinding:
    """Return the artifact binding for ``module``."""

    for module_type in type(module).mro():
        binding = _MODULES.get(cast('type[mxnn.Module]', module_type))
        if binding is not None:
            return binding
    raise ValueError(
        f'{type(module).__module__}.{type(module).__qualname__} is not a '
        'serializable mlx-lattice module.'
    )


def validate_node_against_artifact(node: IRNode) -> None:
    """Validate an IR node against the MLX artifact binding."""

    binding = operation_binding(node.op)
    spec = binding.spec
    _require_keys(node.inputs, spec.inputs, f'{node.id}.inputs')
    _require_keys(node.outputs, spec.outputs, f'{node.id}.outputs')
    allowed = spec.parameters | spec.optional_parameters
    missing = spec.parameters - set(node.parameters)
    extra = set(node.parameters) - allowed
    if missing:
        raise ValueError(
            f'{node.id}.parameters missing required keys: '
            f'{sorted(missing)}.'
        )
    if extra:
        raise ValueError(
            f'{node.id}.parameters has unsupported keys: {sorted(extra)}.'
        )
    allowed_attributes = spec.attributes | spec.value_attributes
    extra_attributes = set(node.attributes) - allowed_attributes
    if extra_attributes:
        raise ValueError(
            f'{node.id}.attributes has unsupported keys: '
            f'{sorted(extra_attributes)}.'
        )
    for name in binding.value_attribute_arguments:
        if name in node.attributes and not isinstance(
            node.attributes[name], str
        ):
            raise ValueError(
                f'{node.id}.attributes.{name} must be a graph value name.'
            )
    for name in binding.attribute_arguments:
        if name in node.attributes:
            _validate_json_attribute(
                node.attributes[name],
                f'{node.id}.attributes.{name}',
            )


def _parameter_bindings(
    bindings: Mapping[str, str | ParameterBinding],
) -> dict[str, ParameterBinding]:
    out: dict[str, ParameterBinding] = {}
    for name, binding in bindings.items():
        out[name] = (
            ParameterBinding(binding)
            if isinstance(binding, str)
            else binding
        )
    return out


def _argument_types(
    function: Callable[..., GraphValue],
    arguments: Mapping[str, str],
) -> dict[str, IRValueType]:
    signature = inspect.signature(function)
    out: dict[str, IRValueType] = {}
    for port, argument in arguments.items():
        parameter = signature.parameters.get(argument)
        if parameter is None:
            continue
        value_type = annotation_value_type(parameter.annotation)
        if value_type != 'any':
            out[port] = value_type
    return out


def _require_keys(
    values: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = set(values)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(
            f'{path} missing required keys: {sorted(missing)}.'
        )
    if extra:
        raise ValueError(f'{path} has unsupported keys: {sorted(extra)}.')


def _validate_json_attribute(value: Any, path: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json_attribute(item, f'{path}[{index}]')
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{path} keys must be strings.')
            _validate_json_attribute(item, f'{path}.{key}')
        return
    raise ValueError(f'{path} must be JSON-compatible.')


def _register_modules() -> None:
    for module_type, spec in _public_module_artifact_specs().items():
        if module_type in _MODULES:
            raise ValueError(
                f'duplicate lattice module binding: {module_type!r}.'
            )
        _MODULES[module_type] = ModuleBinding(
            module_type=module_type,
            op=spec.op,
            parameter_names=spec.parameters,
            attributes=spec.attribute_values,
        )


def _public_module_artifact_specs() -> dict[
    type[mxnn.Module], ModuleArtifactSpec
]:
    out: dict[type[mxnn.Module], ModuleArtifactSpec] = {}
    for name in lnn.__all__:
        module_type = getattr(lnn, name)
        if not inspect.isclass(module_type):
            continue
        spec = module_artifact_spec(cast(type[mxnn.Module], module_type))
        if spec is not None:
            out[cast(type[mxnn.Module], module_type)] = spec
    return out


register_operations(_OperationRegistrar())
_register_modules()
