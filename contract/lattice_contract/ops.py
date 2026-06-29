from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast

from lattice_contract.manifest import IRInputRef, IRNode, IRValueType

DeclarationT = TypeVar('DeclarationT', bound=Callable)
FunctionT = TypeVar('FunctionT', bound=Callable)
IRParameterKind = Literal[
    'array',
    'optional_array',
    'quantized_weight',
    'array_or_quantized_weight',
]


@dataclass(frozen=True, slots=True)
class IROpSpec:
    """Static semantic contract for one lattice IR operation."""

    name: str
    inputs: frozenset[str]
    outputs: frozenset[str]
    output_types: dict[str, IRValueType]
    input_types: dict[str, IRValueType] = field(default_factory=dict)
    value_attribute_types: dict[str, IRValueType] = field(
        default_factory=dict
    )
    parameters: frozenset[str] = frozenset()
    optional_parameters: frozenset[str] = frozenset()
    attributes: frozenset[str] = frozenset()
    value_attributes: frozenset[str] = frozenset()
    requires_support: bool = False


@dataclass(frozen=True, slots=True)
class IROpArtifactHints:
    """Artifact hints that cannot be inferred from annotations alone."""

    parameters: Mapping[str, IRParameterKind] = field(default_factory=dict)
    optional_parameters: Mapping[str, IRParameterKind] = field(
        default_factory=dict
    )
    attributes: frozenset[str] = frozenset()
    value_attributes: frozenset[str] = frozenset()


_OP_SPECS: dict[str, IROpSpec] = {}
_ARTIFACT_HINT_ATTR = '__mlx_lattice_op_artifact_hints__'


def lattice_op_hints(
    *,
    parameters: Mapping[str, str] | None = None,
    optional_parameters: Mapping[str, str] | None = None,
    attributes: set[str] | None = None,
    value_attributes: set[str] | None = None,
) -> Callable[[FunctionT], FunctionT]:
    """Attach artifact classification hints to a public op function.

    Most operation bindings are inferred from annotations. Hints are reserved
    for ambiguous tensor arguments, especially persisted weights and optional
    graph-carried values whose annotations include more than one IR type.
    """

    hints = IROpArtifactHints(
        parameters=_parameter_kinds(parameters),
        optional_parameters=_parameter_kinds(optional_parameters),
        attributes=frozenset(attributes or ()),
        value_attributes=frozenset(value_attributes or ()),
    )

    def decorator(function: FunctionT) -> FunctionT:
        setattr(function, _ARTIFACT_HINT_ATTR, hints)
        return function

    return decorator


def op_artifact_hints(function: Callable) -> IROpArtifactHints:
    """Return artifact hints attached by :func:`lattice_op_hints`."""

    return cast(
        IROpArtifactHints,
        getattr(function, _ARTIFACT_HINT_ATTR, IROpArtifactHints()),
    )


def _parameter_kinds(
    values: Mapping[str, str] | None,
) -> dict[str, IRParameterKind]:
    return {
        name: _parameter_kind(value)
        for name, value in dict(values or {}).items()
    }


def _parameter_kind(value: str) -> IRParameterKind:
    if value not in (
        'array',
        'optional_array',
        'quantized_weight',
        'array_or_quantized_weight',
    ):
        raise ValueError(f'unsupported IR parameter kind: {value!r}.')
    return cast('IRParameterKind', value)


def ir_op_spec(
    name: str,
    *,
    inputs: set[str],
    outputs: set[str],
    output_types: Mapping[str, IRValueType] | None = None,
    input_types: Mapping[str, IRValueType] | None = None,
    value_attribute_types: Mapping[str, IRValueType] | None = None,
    parameters: set[str] | None = None,
    optional_parameters: set[str] | None = None,
    attributes: set[str] | None = None,
    value_attributes: set[str] | None = None,
    requires_support: bool = False,
) -> Callable[[DeclarationT], DeclarationT]:
    """Register the semantic contract for one IR operation.

    The annotation keeps the stable IR operation set compact and discoverable
    without maintaining a hand-written registry table. Runtime backends attach
    their implementations separately, so importing :mod:`mlx_lattice.ir`
    always exposes the complete semantic contract without importing a backend
    graph executor.
    """

    spec = IROpSpec(
        name=name,
        inputs=frozenset(inputs),
        outputs=frozenset(outputs),
        output_types=dict(output_types or {}),
        input_types=dict(input_types or {}),
        value_attribute_types=dict(value_attribute_types or {}),
        parameters=frozenset(parameters or ()),
        optional_parameters=frozenset(optional_parameters or ()),
        attributes=frozenset(attributes or ()),
        value_attributes=frozenset(value_attributes or ()),
        requires_support=requires_support,
    )

    def decorator(declaration: DeclarationT) -> DeclarationT:
        if name in _OP_SPECS:
            raise ValueError(
                f'duplicate lattice IR op registration: {name}.'
            )
        _OP_SPECS[name] = spec
        return declaration

    return decorator


def iter_op_specs() -> Iterator[IROpSpec]:
    """Iterate registered IR operation specs."""

    return iter(_OP_SPECS.values())


def op_spec(name: str) -> IROpSpec:
    """Return the registered spec for ``name`` or raise ``ValueError``."""

    try:
        return _OP_SPECS[name]
    except KeyError:
        pass
    raise ValueError(f'unsupported lattice IR op: {name!r}.')


def validate_node_against_spec(node: IRNode) -> None:
    """Validate node ports/parameters against the registered op spec."""

    spec = op_spec(node.op)
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
    if spec.requires_support and node.support is None:
        raise ValueError(f'{node.id} requires a support object.')


def _require_keys(
    values: Mapping[str, IRInputRef],
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
