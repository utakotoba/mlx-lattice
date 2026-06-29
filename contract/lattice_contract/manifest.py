from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

CURRENT_SCHEMA_VERSION = '0.1'

DTypePolicy = Literal['preserve', 'fp32', 'fp16', 'fp16_inference']
IRValueType = Literal[
    'any',
    'sparse_tensor',
    'dense_tensor',
    'relation',
    'coordinate_set',
    'alignment',
    'quantization',
    'point_voxel_map',
    'coordinate_ordering',
    'sparse_occupancy',
    'occupancy_expansion',
    'bytes',
]
IRParameter = str
type IRInputRef = str | tuple[str, ...]
Triple = tuple[int, int, int]

_IR_VALUE_TYPES = frozenset(
    (
        'any',
        'sparse_tensor',
        'dense_tensor',
        'relation',
        'coordinate_set',
        'alignment',
        'quantization',
        'point_voxel_map',
        'coordinate_ordering',
        'sparse_occupancy',
        'occupancy_expansion',
        'bytes',
    )
)
_COORDINATE_ORDER = ('batch', 'x', 'y', 'z')
_FEATURE_LAYOUT = ('row', 'channel')
_WEIGHT_LAYOUT = 'mlx-lattice'


def triple(value: int | Sequence[int], *, name: str) -> Triple:
    """Normalize an integer or 3-sequence into a spatial integer triple."""

    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f'{name} must be an int or a sequence of 3 ints.')
    out = tuple(int(item) for item in value)
    if len(out) != 3:
        raise ValueError(f'{name} must be an int or a sequence of 3 ints.')
    return cast('Triple', out)


@dataclass(frozen=True, slots=True)
class IRSparseSupport:
    """Sparse relation/support attributes for an IR operation."""

    kind: str
    kernel_size: Triple | None = None
    stride: Triple | None = None
    padding: Triple | None = None
    dilation: Triple | None = None
    target: str | None = None
    mode: str | None = None
    join: str | None = None


@dataclass(frozen=True, slots=True)
class IRTensorSpec:
    """Named graph input or output specification."""

    name: str
    type: IRValueType


@dataclass(frozen=True, slots=True)
class IRNode:
    """One semantic operation in a lattice model graph."""

    id: str
    op: str
    inputs: dict[str, IRInputRef]
    outputs: dict[str, str]
    parameters: dict[str, IRParameter] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    support: IRSparseSupport | None = None


@dataclass(frozen=True, slots=True)
class IRManifest:
    """Validated sparse model manifest.

    The manifest is the stable artifact contract shared by future training
    exporters and the MLX runtime loader. It records semantic sparse graph
    nodes and names the tensor weights stored beside it.
    """

    schema_version: str
    producer: dict[str, str]
    runtime: dict[str, str]
    inputs: tuple[IRTensorSpec, ...]
    outputs: tuple[IRTensorSpec, ...]
    nodes: tuple[IRNode, ...]
    dtype_policy: DTypePolicy = 'preserve'
    coordinate_order: tuple[str, ...] = _COORDINATE_ORDER
    feature_layout: tuple[str, ...] = _FEATURE_LAYOUT
    weight_layout: str = _WEIGHT_LAYOUT


def load_manifest(path: str | Path) -> IRManifest:
    """Load and validate a manifest JSON file."""

    with Path(path).open('r', encoding='utf-8') as file:
        raw = json.load(file)
    return manifest_from_dict(raw)


def manifest_from_dict(raw: Mapping[str, Any]) -> IRManifest:
    """Build a validated :class:`IRManifest` from decoded JSON data."""

    _require_mapping(raw, 'manifest')
    schema_version = _require_str(raw, 'schema_version')
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f'unsupported lattice IR schema_version {schema_version!r}; '
            f'expected {CURRENT_SCHEMA_VERSION!r}.'
        )

    coordinate_order = _str_tuple(
        raw.get('coordinate_order', _COORDINATE_ORDER),
        'coordinate_order',
    )
    if coordinate_order != _COORDINATE_ORDER:
        raise ValueError(
            "coordinate_order must be ['batch', 'x', 'y', 'z']."
        )

    feature_layout = _str_tuple(
        raw.get('feature_layout', _FEATURE_LAYOUT),
        'feature_layout',
    )
    if feature_layout != _FEATURE_LAYOUT:
        raise ValueError("feature_layout must be ['row', 'channel'].")

    weight_layout = _require_str(
        raw, 'weight_layout', default=_WEIGHT_LAYOUT
    )
    if weight_layout != _WEIGHT_LAYOUT:
        raise ValueError("weight_layout must be 'mlx-lattice'.")

    dtype_policy = _dtype_policy(
        _require_str(raw, 'dtype_policy', default='preserve')
    )

    manifest = IRManifest(
        schema_version=schema_version,
        producer=_str_map(raw.get('producer', {}), 'producer'),
        runtime=_str_map(raw.get('runtime', {}), 'runtime'),
        coordinate_order=coordinate_order,
        feature_layout=feature_layout,
        weight_layout=weight_layout,
        dtype_policy=dtype_policy,
        inputs=tuple(
            _tensor_spec(item, f'inputs[{index}]')
            for index, item in enumerate(_require_list(raw, 'inputs'))
        ),
        outputs=tuple(
            _tensor_spec(item, f'outputs[{index}]')
            for index, item in enumerate(_require_list(raw, 'outputs'))
        ),
        nodes=tuple(
            _node(item, f'nodes[{index}]')
            for index, item in enumerate(_require_list(raw, 'nodes'))
        ),
    )
    _validate_graph(manifest)
    return manifest


def manifest_to_dict(manifest: IRManifest) -> dict[str, Any]:
    """Convert a manifest object to JSON-serializable data."""

    return {
        'schema_version': manifest.schema_version,
        'producer': dict(manifest.producer),
        'runtime': dict(manifest.runtime),
        'coordinate_order': list(manifest.coordinate_order),
        'feature_layout': list(manifest.feature_layout),
        'weight_layout': manifest.weight_layout,
        'dtype_policy': manifest.dtype_policy,
        'inputs': [
            {'name': item.name, 'type': item.type}
            for item in manifest.inputs
        ],
        'outputs': [
            {'name': item.name, 'type': item.type}
            for item in manifest.outputs
        ],
        'nodes': [_node_to_dict(node) for node in manifest.nodes],
    }


def is_ir_value_type(value: str) -> bool:
    """Return whether ``value`` is a supported lattice IR value type."""

    return value in _IR_VALUE_TYPES


def ir_value_type(value: str) -> IRValueType:
    """Validate and cast a string into :class:`IRValueType`."""

    if not is_ir_value_type(value):
        raise ValueError(f'unsupported IR value type: {value!r}.')
    return cast('IRValueType', value)


def _node_to_dict(node: IRNode) -> dict[str, Any]:
    raw: dict[str, Any] = {
        'id': node.id,
        'op': node.op,
        'inputs': {
            key: value if isinstance(value, str) else list(value)
            for key, value in node.inputs.items()
        },
        'outputs': dict(node.outputs),
    }
    if node.parameters:
        raw['parameters'] = dict(node.parameters)
    if node.attributes:
        raw['attributes'] = dict(node.attributes)
    if node.support is not None:
        raw['support'] = _support_to_dict(node.support)
    return raw


def _support_to_dict(support: IRSparseSupport) -> dict[str, Any]:
    raw: dict[str, Any] = {'kind': support.kind}
    for name in ('kernel_size', 'stride', 'padding', 'dilation'):
        value = getattr(support, name)
        if value is not None:
            raw[name] = list(value)
    for name in ('target', 'mode', 'join'):
        value = getattr(support, name)
        if value is not None:
            raw[name] = value
    return raw


def _support_attributes(support: IRSparseSupport) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for name in ('kernel_size', 'stride', 'padding', 'dilation'):
        value = getattr(support, name)
        if value is not None:
            raw[name] = value
    if support.target is not None:
        raw['coordinates'] = support.target
    if support.mode is not None:
        raw['mode'] = support.mode
    if support.join is not None:
        raw['join'] = support.join
    return raw


def _node(raw: Any, path: str) -> IRNode:
    data = _require_mapping(raw, path)
    attributes = dict(
        _require_mapping(
            data.get('attributes', {}),
            f'{path}.attributes',
        )
    )
    support = (
        None
        if 'support' not in data
        else _support(data['support'], f'{path}.support')
    )
    if support is not None:
        attributes.update(_support_attributes(support))
    return IRNode(
        id=_require_str(data, 'id', path=path),
        op=_require_str(data, 'op', path=path),
        inputs=_input_map(data.get('inputs', {}), f'{path}.inputs'),
        outputs=_str_map(data.get('outputs', {}), f'{path}.outputs'),
        parameters=_str_map(
            data.get('parameters', {}), f'{path}.parameters'
        ),
        attributes=attributes,
        support=support,
    )


def _support(raw: Any, path: str) -> IRSparseSupport:
    data = _require_mapping(raw, path)
    return IRSparseSupport(
        kind=_require_str(data, 'kind', path=path),
        kernel_size=_optional_triple(data, 'kernel_size', path),
        stride=_optional_triple(data, 'stride', path),
        padding=_optional_triple(data, 'padding', path),
        dilation=_optional_triple(data, 'dilation', path),
        target=_optional_str(data, 'target', path),
        mode=_optional_str(data, 'mode', path),
        join=_optional_str(data, 'join', path),
    )


def _tensor_spec(raw: Any, path: str) -> IRTensorSpec:
    data = _require_mapping(raw, path)
    name = _require_str(data, 'name', path=path)
    value_type = _require_str(data, 'type', path=path)
    try:
        typed = ir_value_type(value_type)
    except ValueError as exc:
        raise ValueError(
            f'{path}.type is not a supported lattice IR value type.'
        ) from exc
    return IRTensorSpec(name, typed)


def _validate_graph(manifest: IRManifest) -> None:
    names: set[str] = set()
    node_ids: set[str] = set()
    for item in manifest.inputs:
        _require_unique(names, item.name, 'graph value')
    for node in manifest.nodes:
        _require_unique(node_ids, node.id, 'node id')
        if not node.outputs:
            raise ValueError(f'node {node.id!r} must define outputs.')
        for value_ref in node.inputs.values():
            for value_name in _value_refs(value_ref):
                if value_name not in names:
                    raise ValueError(
                        f'node {node.id!r} references unknown input '
                        f'{value_name!r}.'
                    )
        for value_name in node.outputs.values():
            _require_unique(names, value_name, 'graph value')
    for item in manifest.outputs:
        if item.name not in names:
            raise ValueError(f'unknown manifest output {item.name!r}.')


def _require_unique(seen: set[str], value: str, label: str) -> None:
    if value in seen:
        raise ValueError(f'duplicate {label}: {value!r}.')
    seen.add(value)


def _optional_triple(
    data: Mapping[str, Any],
    name: str,
    path: str,
) -> Triple | None:
    if name not in data:
        return None
    return triple(data[name], name=f'{path}.{name}')


def _dtype_policy(value: str) -> DTypePolicy:
    if value not in ('preserve', 'fp32', 'fp16', 'fp16_inference'):
        raise ValueError(
            'dtype_policy must be preserve, fp32, fp16, or fp16_inference.'
        )
    return cast('DTypePolicy', value)


def _str_tuple(raw: Any, path: str) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f'{path} must be a list of strings.')
    values = tuple(raw)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f'{path} must be a list of strings.')
    return cast(tuple[str, ...], values)


def _str_map(raw: Any, path: str) -> dict[str, str]:
    data = _require_mapping(raw, path)
    out: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f'{path} must map strings to strings.')
        out[key] = value
    return out


def _input_map(raw: Any, path: str) -> dict[str, IRInputRef]:
    data = _require_mapping(raw, path)
    out: dict[str, IRInputRef] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f'{path} must map strings to value refs.')
        if isinstance(value, str):
            out[key] = value
            continue
        if isinstance(value, list | tuple) and all(
            isinstance(item, str) for item in value
        ):
            out[key] = tuple(value)
            continue
        raise ValueError(
            f'{path}.{key} must be a string or list of strings.'
        )
    return out


def _value_refs(value: IRInputRef) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _require_mapping(raw: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f'{path} must be an object.')
    return raw


def _require_list(raw: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = raw.get(name)
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list.')
    return value


def _require_str(
    raw: Mapping[str, Any],
    name: str,
    *,
    path: str = '',
    default: str | None = None,
) -> str:
    value = raw.get(name, default)
    if not isinstance(value, str):
        prefix = f'{path}.' if path else ''
        raise ValueError(f'{prefix}{name} must be a string.')
    return value


def _optional_str(
    raw: Mapping[str, Any],
    name: str,
    path: str,
) -> str | None:
    if name not in raw:
        return None
    value = raw[name]
    if not isinstance(value, str):
        raise ValueError(f'{path}.{name} must be a string.')
    return value
