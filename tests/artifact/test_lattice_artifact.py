from __future__ import annotations

import inspect
from typing import Protocol, cast

import mlx.nn as mxnn
import pytest
from lattice_contract import (
    IRValueType,
    manifest_from_dict,
    manifest_to_dict,
)

from mlx_lattice import SparseTensor
from mlx_lattice import nn as lnn
from mlx_lattice import ops as lops
from mlx_lattice.artifact import (
    GraphOutput,
    LatticeArtifact,
    LatticeGraphBuilder,
    LatticeModel,
    build_lattice_graph_artifact,
    build_lattice_module_artifact,
    iter_operation_specs,
    load_lattice_artifact,
    load_lattice_model,
    module_artifact_binding,
    operation_binding,
    save_lattice_graph,
    save_lattice_model,
    save_lattice_module,
)
from mlx_lattice.artifact.bindings import value_type_fields
from mlx_lattice.nn._artifact import module_artifact_spec
from mlx_lattice.ops import (
    conv3d,
    global_avg_pool,
    kernel_offsets,
    linear,
    morton_sort_coords,
    occupancy_downsample,
    occupancy_expand,
    relu,
    sparse_binary_op,
    topk_rows,
    voxelize,
)
from tests.support import assert_nested_close, mx

pytestmark = [pytest.mark.usefixtures('selected_backend')]


class _ActiveRowsValue(Protocol):
    active_rows: mx.array


def _input() -> SparseTensor:
    return SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        batch_counts=(3,),
    )


def _manifest():
    return manifest_from_dict(
        {
            'schema_version': '0.1',
            'producer': {'name': 'test'},
            'runtime': {'name': 'mlx-lattice', 'version': '>=0.2,<0.3'},
            'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
            'outputs': [{'name': 'logits', 'type': 'dense_tensor'}],
            'nodes': [
                {
                    'id': 'conv',
                    'op': 'sparse.conv3d',
                    'inputs': {'input': 'input'},
                    'outputs': {'output': 'conv'},
                    'parameters': {'weight': 'conv.weight'},
                    'support': {
                        'kind': 'forward',
                        'kernel_size': [3, 1, 1],
                        'stride': [1, 1, 1],
                        'padding': [0, 0, 0],
                        'dilation': [1, 1, 1],
                    },
                },
                {
                    'id': 'relu',
                    'op': 'feature.relu',
                    'inputs': {'input': 'conv'},
                    'outputs': {'output': 'relu'},
                },
                {
                    'id': 'linear',
                    'op': 'feature.linear',
                    'inputs': {'input': 'relu'},
                    'outputs': {'output': 'projected'},
                    'parameters': {
                        'weight': 'linear.weight',
                        'bias': 'linear.bias',
                    },
                },
                {
                    'id': 'pool',
                    'op': 'pool.global_avg',
                    'inputs': {'input': 'projected'},
                    'outputs': {'output': 'logits'},
                },
            ],
        }
    )


def _weights():
    return {
        'conv.weight': mx.array([1.0, 2.0, 3.0], dtype=mx.float32).reshape(
            1, 3, 1, 1, 1
        ),
        'linear.weight': mx.array([[2.0], [3.0]], dtype=mx.float32),
        'linear.bias': mx.array([1.0, -1.0], dtype=mx.float32),
    }


def _module_instance(name: str):
    if name in {
        'Conv3d',
        'ConvTranspose3d',
        'GenerativeConvTranspose3d',
        'QuantizedConv3d',
        'QuantizedConvTranspose3d',
        'QuantizedGenerativeConvTranspose3d',
    }:
        return getattr(lnn, name)(1, 1)
    if name in {'SubmConv3d', 'QuantizedSubmConv3d'}:
        return getattr(lnn, name)(1, 1, kernel_size=1)
    if name in {'Linear', 'QuantizedLinear'}:
        return getattr(lnn, name)(1, 1)
    if name in {'BatchNorm', 'LayerNorm', 'RMSNorm'}:
        return getattr(lnn, name)(1)
    return getattr(lnn, name)()


def _active_rows(value: _ActiveRowsValue) -> int:
    mx.eval(value.active_rows)
    return int(value.active_rows.tolist()[0])


def _active_prefix(value: mx.array, rows: int) -> mx.array:
    return value[:rows]


def _assert_active_array_equal(
    actual: mx.array,
    expected: mx.array,
    rows: int,
) -> None:
    actual_prefix = _active_prefix(actual, rows)
    expected_prefix = _active_prefix(expected, rows)
    mx.eval(actual_prefix, expected_prefix)
    assert actual_prefix.tolist() == expected_prefix.tolist()


def test_lattice_model_runs_manifest_graph_with_public_ops() -> None:
    x = _input()
    model = LatticeModel(_manifest(), _weights())

    actual = model(x)
    expected_sparse = linear(
        relu(conv3d(x, _weights()['conv.weight'], kernel_size=(3, 1, 1))),
        _weights()['linear.weight'],
        _weights()['linear.bias'],
    )
    expected = mx.mean(expected_sparse.feats, axis=0, keepdims=True)
    mx.eval(actual, expected)

    assert_nested_close(actual.tolist(), expected.tolist())


def test_lattice_artifact_roundtrips_through_safetensors(tmp_path) -> None:
    x = _input()
    save_lattice_model(tmp_path, _manifest(), _weights())

    artifact = load_lattice_artifact(tmp_path)
    model = load_lattice_model(tmp_path)
    actual = model(x)
    mx.eval(actual)

    assert isinstance(artifact, LatticeArtifact)
    assert isinstance(artifact.model(), LatticeModel)
    assert actual.shape == (1, 2)


def test_lattice_model_accepts_compatible_runtime_metadata() -> None:
    raw = manifest_to_dict(_manifest())
    raw['runtime'] = {'name': 'mlx-lattice', 'version': '>=0.2.0,<0.3.0'}
    manifest = manifest_from_dict(raw)

    model = LatticeModel(manifest, _weights())

    assert model.manifest.runtime == raw['runtime']


def test_lattice_model_accepts_missing_runtime_metadata() -> None:
    raw = manifest_to_dict(_manifest())
    raw['runtime'] = {}
    manifest = manifest_from_dict(raw)

    model = LatticeModel(manifest, _weights())

    assert model.manifest.runtime == {}


def test_lattice_model_rejects_incompatible_runtime_name() -> None:
    raw = manifest_to_dict(_manifest())
    raw['runtime'] = {'name': 'other-runtime', 'version': '>=0.2,<0.3'}
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match=r'runtime\.name'):
        LatticeModel(manifest, _weights())


def test_lattice_model_rejects_incompatible_runtime_version() -> None:
    raw = manifest_to_dict(_manifest())
    raw['runtime'] = {'name': 'mlx-lattice', 'version': '>=999.0'}
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match=r'runtime\.version'):
        LatticeModel(manifest, _weights())


def test_lattice_model_rejects_malformed_runtime_version_specifier() -> (
    None
):
    raw = manifest_to_dict(_manifest())
    raw['runtime'] = {'name': 'mlx-lattice', 'version': '~=0.2'}
    manifest = manifest_from_dict(raw)

    with pytest.raises(
        ValueError, match='unsupported runtime version specifier'
    ):
        LatticeModel(manifest, _weights())


def test_lattice_model_rejects_unknown_operation() -> None:
    raw = {
        'schema_version': '0.1',
        'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
        'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'bad',
                'op': 'custom.bad',
                'inputs': {'input': 'input'},
                'outputs': {'output': 'output'},
            }
        ],
    }
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match='unsupported lattice IR op'):
        LatticeModel(manifest, {})


def test_lattice_model_rejects_unknown_value_attribute_reference() -> None:
    raw = {
        'schema_version': '0.1',
        'inputs': [
            {'name': 'points', 'type': 'dense_tensor'},
            {'name': 'feats', 'type': 'dense_tensor'},
        ],
        'outputs': [{'name': 'voxels', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'voxelize',
                'op': 'ops.voxelize',
                'inputs': {'points': 'points', 'feats': 'feats'},
                'outputs': {'output': 'voxels'},
                'attributes': {'batch_indices': 'missing'},
            }
        ],
    }
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match='unknown graph value'):
        LatticeModel(manifest, {})


def test_lattice_model_rejects_static_input_type_mismatch() -> None:
    raw = {
        'schema_version': '0.1',
        'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
        'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'bad',
                'op': 'ops.voxelize',
                'inputs': {'points': 'input', 'feats': 'input'},
                'outputs': {'output': 'output'},
            }
        ],
    }
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match=r"expects 'dense_tensor'"):
        LatticeModel(manifest, {})


def test_lattice_model_rejects_static_value_attribute_type_mismatch() -> (
    None
):
    raw = {
        'schema_version': '0.1',
        'inputs': [
            {'name': 'points', 'type': 'dense_tensor'},
            {'name': 'feats', 'type': 'dense_tensor'},
            {'name': 'bad_map', 'type': 'dense_tensor'},
        ],
        'outputs': [{'name': 'sampled', 'type': 'dense_tensor'}],
        'nodes': [
            {
                'id': 'voxelize',
                'op': 'ops.voxelize',
                'inputs': {'points': 'points', 'feats': 'feats'},
                'outputs': {'output': 'voxels'},
            },
            {
                'id': 'sample',
                'op': 'ops.devoxelize',
                'inputs': {'points': 'points', 'voxels': 'voxels'},
                'outputs': {'output': 'sampled'},
                'attributes': {'point_voxel_map': 'bad_map'},
            },
        ],
    }
    manifest = manifest_from_dict(raw)

    with pytest.raises(ValueError, match=r"expects 'point_voxel_map'"):
        LatticeModel(manifest, {})


def test_lattice_model_rejects_missing_weight_payload() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match='missing weight'):
        LatticeModel(manifest, {})


def test_lattice_artifact_writer_rejects_missing_weight_payload(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match='missing weight'):
        save_lattice_model(tmp_path, _manifest(), {})


def test_lattice_model_applies_fp16_dtype_policy_to_sparse_values_and_weights() -> (
    None
):
    raw = {
        'schema_version': '0.1',
        'dtype_policy': 'fp16',
        'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
        'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'linear',
                'op': 'feature.linear',
                'inputs': {'input': 'input'},
                'outputs': {'output': 'output'},
                'parameters': {
                    'weight': 'linear.weight',
                    'bias': 'linear.bias',
                },
            }
        ],
    }
    manifest = manifest_from_dict(raw)
    model = LatticeModel(manifest, _weights())

    actual = model(_input())
    mx.eval(actual.feats)

    assert model.weights['linear.weight'].dtype == mx.float16
    assert model.weights['linear.bias'].dtype == mx.float16
    assert actual.feats.dtype == mx.float16


def test_lattice_model_preserves_quantized_payload_dtypes_under_dtype_policy() -> (
    None
):
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    artifact = build_lattice_module_artifact(dense.to_quantized(bits=4))
    raw = manifest_to_dict(artifact.manifest)
    raw['dtype_policy'] = 'fp16'
    manifest = manifest_from_dict(raw)

    model = LatticeModel(manifest, artifact.weights)
    prefix = manifest.nodes[0].parameters['weight']

    assert model.weights[f'{prefix}.weight'].dtype == mx.uint32
    assert (
        model.weights[f'{prefix}.scales'].dtype
        == artifact.weights[f'{prefix}.scales'].dtype
    )
    assert (
        model.weights[f'{prefix}.biases'].dtype
        == artifact.weights[f'{prefix}.biases'].dtype
    )


def test_lattice_model_rejects_incomplete_quantized_weight_payload() -> (
    None
):
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    artifact = build_lattice_module_artifact(dense.to_quantized(bits=4))
    prefix = artifact.manifest.nodes[0].parameters['weight']
    weights = dict(artifact.weights)
    del weights[f'{prefix}.attrs']

    with pytest.raises(ValueError, match='missing quantized weight'):
        LatticeModel(artifact.manifest, weights)


def test_lattice_model_supports_target_convolution() -> None:
    x = _input()
    target = mx.array([[0, 1, 0, 0], [0, 3, 0, 0]], dtype=mx.int32)
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [
                {'name': 'input', 'type': 'sparse_tensor'},
                {'name': 'target', 'type': 'dense_tensor'},
            ],
            'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'target_conv',
                    'op': 'sparse.conv3d',
                    'inputs': {'input': 'input'},
                    'outputs': {'output': 'output'},
                    'parameters': {'weight': 'conv.weight'},
                    'support': {
                        'kind': 'target',
                        'target': 'target',
                        'kernel_size': [3, 1, 1],
                    },
                }
            ],
        }
    )
    model = LatticeModel(manifest, _weights())

    actual = model(x, target)
    expected = conv3d(
        x,
        _weights()['conv.weight'],
        kernel_size=(3, 1, 1),
        coordinates=target,
    )
    mx.eval(actual.feats, expected.feats)

    assert actual.coords.shape == expected.coords.shape
    assert mx.array_equal(actual.coords, expected.coords).item()
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_lattice_runtime_registry_covers_public_ops_surface() -> None:
    names = {spec.name for spec in iter_operation_specs()}

    assert 'ops.voxelize' in names
    assert 'ops.devoxelize' in names
    assert 'ops.kernel_relation' in names
    assert 'ops.range_encode' in names
    assert 'sparse.quantized_conv3d' in names
    assert 'feature.quantized_linear' in names
    assert not any(name.startswith('ops.Sparse') for name in names)
    assert operation_binding('ops.voxelize').spec.output_types == {
        'output': 'sparse_tensor'
    }
    assert operation_binding('ops.voxelize').spec.input_types == {
        'points': 'dense_tensor',
        'feats': 'dense_tensor',
    }
    assert operation_binding(
        'ops.devoxelize'
    ).spec.value_attribute_types == {
        'batch_indices': 'dense_tensor',
        'point_active_rows': 'dense_tensor',
        'point_voxel_map': 'point_voxel_map',
    }
    assert operation_binding('pool.global_avg').spec.output_types == {
        'output': 'dense_tensor'
    }
    assert operation_binding('ops.voxelize').value_attribute_arguments == {
        'active_rows': 'active_rows',
        'batch_indices': 'batch_indices',
    }


def test_lattice_runtime_registry_keeps_tensors_out_of_json_attributes() -> (
    None
):
    batch_norm = operation_binding('feature.batch_norm').spec
    layer_norm = operation_binding('feature.layer_norm').spec
    rms_norm = operation_binding('feature.rms_norm').spec

    assert batch_norm.optional_parameters == frozenset(
        {'weight', 'bias', 'mean', 'var'}
    )
    assert batch_norm.attributes == frozenset({'eps'})
    assert layer_norm.optional_parameters == frozenset({'weight', 'bias'})
    assert layer_norm.attributes == frozenset({'eps'})
    assert rms_norm.optional_parameters == frozenset({'weight'})
    assert rms_norm.attributes == frozenset({'eps'})


def test_lattice_runtime_registry_uses_variant_specific_quantized_conv_contracts() -> (
    None
):
    assert operation_binding(
        'sparse.quantized_subm_conv3d'
    ).spec.attributes == frozenset({'kernel_size', 'dilation'})
    assert operation_binding(
        'sparse.quantized_generative_conv_transpose3d'
    ).spec.attributes == frozenset({'kernel_size', 'stride'})
    assert (
        operation_binding(
            'sparse.quantized_conv_transpose3d'
        ).spec.value_attributes
        == frozenset()
    )


def test_lattice_runtime_registry_tracks_public_ops_functions() -> None:
    names = {spec.name for spec in iter_operation_specs()}
    public_functions = {
        name
        for name in lops.__all__
        if inspect.isfunction(getattr(lops, name))
    }

    missing = {
        name for name in public_functions if f'ops.{name}' not in names
    }
    stale = {
        name.removeprefix('ops.')
        for name in names
        if name.startswith('ops.')
        and name.removeprefix('ops.') not in public_functions
    }

    assert missing == set()
    assert stale == set()


def test_lattice_runtime_registry_keeps_generic_and_semantic_op_routes() -> (
    None
):
    names = {spec.name for spec in iter_operation_specs()}

    assert 'ops.conv3d' in names
    assert 'sparse.conv3d' in names
    assert 'ops.linear' in names
    assert 'feature.linear' in names
    assert operation_binding('ops.linear').spec.parameters == frozenset(
        {'weight'}
    )
    assert operation_binding(
        'ops.linear'
    ).spec.optional_parameters == frozenset({'bias'})
    assert operation_binding(
        'ops.conv3d'
    ).spec.value_attributes == frozenset({'coordinates'})


def test_lattice_module_artifact_registry_tracks_public_nn_modules() -> (
    None
):
    modules = {
        name: getattr(lnn, name)
        for name in lnn.__all__
        if inspect.isclass(getattr(lnn, name))
    }
    expected_ops = {
        'AvgPool3d': 'ops.avg_pool3d',
        'BatchNorm': 'feature.batch_norm',
        'Conv3d': 'sparse.conv3d',
        'ConvTranspose3d': 'sparse.conv_transpose3d',
        'Dropout': 'feature.dropout',
        'GELU': 'feature.gelu',
        'GenerativeConvTranspose3d': 'sparse.generative_conv_transpose3d',
        'GlobalAvgPool': 'pool.global_avg',
        'GlobalMaxPool': 'pool.global_max',
        'GlobalSumPool': 'pool.global_sum',
        'LayerNorm': 'feature.layer_norm',
        'LeakyReLU': 'feature.leaky_relu',
        'Linear': 'feature.linear',
        'MaxPool3d': 'ops.max_pool3d',
        'Pool3d': 'ops.pool3d',
        'QuantizedConv3d': 'sparse.quantized_conv3d',
        'QuantizedConvTranspose3d': 'sparse.quantized_conv_transpose3d',
        'QuantizedGenerativeConvTranspose3d': (
            'sparse.quantized_generative_conv_transpose3d'
        ),
        'QuantizedLinear': 'feature.quantized_linear',
        'QuantizedSubmConv3d': 'sparse.quantized_subm_conv3d',
        'RMSNorm': 'feature.rms_norm',
        'ReLU': 'feature.relu',
        'SiLU': 'feature.silu',
        'Sigmoid': 'feature.sigmoid',
        'Softplus': 'feature.softplus',
        'SubmConv3d': 'sparse.subm_conv3d',
        'SumPool3d': 'ops.sum_pool3d',
        'Tanh': 'feature.tanh',
    }

    assert set(modules) == set(expected_ops)
    for name, cls in modules.items():
        assert module_artifact_spec(cls) is not None
        instance = _module_instance(name)
        assert isinstance(instance, cls)
        assert module_artifact_binding(instance).op == expected_ops[name]


def test_lattice_model_runs_generic_public_sparse_op() -> None:
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        batch_counts=(3,),
    )
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
            'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'crop',
                    'op': 'ops.crop',
                    'inputs': {'x': 'input'},
                    'outputs': {'output': 'output'},
                    'attributes': {
                        'min_coord': [1, 0, 0],
                        'max_coord': [2, 0, 0],
                    },
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})(x)
    mx.eval(actual.coords, actual.feats)

    assert actual.coords.tolist() == [[0, 1, 0, 0], [0, 2, 0, 0]]
    assert actual.feats.tolist() == [[2.0], [3.0]]


def test_lattice_model_resolves_value_attributes_for_generic_ops() -> None:
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)
    batches = mx.array([0, 0, 0], dtype=mx.int32)
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [
                {'name': 'points', 'type': 'dense_tensor'},
                {'name': 'feats', 'type': 'dense_tensor'},
                {'name': 'batches', 'type': 'dense_tensor'},
            ],
            'outputs': [{'name': 'voxels', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'voxelize',
                    'op': 'ops.voxelize',
                    'inputs': {'points': 'points', 'feats': 'feats'},
                    'outputs': {'output': 'voxels'},
                    'attributes': {
                        'batch_indices': 'batches',
                        'reduction': 'mean',
                        'voxel_size': 1.0,
                    },
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})(points, feats, batches)
    expected = voxelize(
        points,
        feats,
        batch_indices=batches,
        reduction='mean',
        voxel_size=1.0,
    )
    mx.eval(actual.coords, actual.feats, expected.coords, expected.feats)

    assert actual.coords.tolist() == expected.coords.tolist()
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_lattice_model_treats_required_constants_as_attributes() -> None:
    offsets_binding = operation_binding('ops.kernel_offsets')
    binary_binding = operation_binding('ops.sparse_binary_op')
    topk_binding = operation_binding('ops.topk_rows')

    assert offsets_binding.spec.inputs == frozenset()
    assert 'kernel_size' in offsets_binding.spec.attributes
    assert 'kernel_size' in offsets_binding.attribute_arguments
    assert binary_binding.spec.inputs == frozenset({'lhs', 'rhs'})
    assert 'op' in binary_binding.spec.attributes
    assert 'op' in binary_binding.attribute_arguments
    assert topk_binding.spec.inputs == frozenset({'x'})
    assert 'counts' in topk_binding.spec.attributes
    assert 'counts' in topk_binding.attribute_arguments


def test_lattice_model_runs_generic_constant_only_op() -> None:
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [],
            'outputs': [{'name': 'offsets', 'type': 'any'}],
            'nodes': [
                {
                    'id': 'offsets',
                    'op': 'ops.kernel_offsets',
                    'outputs': {'output': 'offsets'},
                    'attributes': {'kernel_size': [1, 1, 1]},
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})()

    assert actual == kernel_offsets((1, 1, 1))


def test_lattice_model_runs_generic_op_with_required_constant() -> None:
    lhs = _input()
    rhs = SparseTensor(
        lhs.coords,
        mx.array([[0.5], [1.5], [2.5]], dtype=mx.float32),
        batch_counts=(3,),
    )
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [
                {'name': 'lhs', 'type': 'sparse_tensor'},
                {'name': 'rhs', 'type': 'sparse_tensor'},
            ],
            'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
            'nodes': [
                {
                    'id': 'binary',
                    'op': 'ops.sparse_binary_op',
                    'inputs': {'lhs': 'lhs', 'rhs': 'rhs'},
                    'outputs': {'output': 'output'},
                    'attributes': {'op': 'sub', 'join': 'inner'},
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})(lhs, rhs)
    expected = sparse_binary_op(lhs, rhs, 'sub', join='inner')
    mx.eval(actual.coords, actual.feats, expected.coords, expected.feats)

    assert actual.coords.tolist() == expected.coords.tolist()
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_lattice_model_runs_generic_sparse_op_with_count_attribute() -> (
    None
):
    x = SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [4.0], [2.0]], dtype=mx.float32),
        batch_counts=(3,),
    )
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
            'outputs': [{'name': 'rows', 'type': 'dense_tensor'}],
            'nodes': [
                {
                    'id': 'topk',
                    'op': 'ops.topk_rows',
                    'inputs': {'x': 'input'},
                    'outputs': {'output': 'rows'},
                    'attributes': {'counts': [2], 'rho': 1.0},
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})(x)
    expected = topk_rows(x, [2], rho=1.0)
    mx.eval(actual, expected)

    assert actual.tolist() == expected.tolist()


def test_lattice_model_supports_typed_coordinate_ordering_output() -> None:
    coords = mx.array(
        [[0, 2, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [{'name': 'coords', 'type': 'dense_tensor'}],
            'outputs': [
                {'name': 'ordering', 'type': 'coordinate_ordering'}
            ],
            'nodes': [
                {
                    'id': 'sort',
                    'op': 'ops.morton_sort_coords',
                    'inputs': {'coords': 'coords'},
                    'outputs': {'output': 'ordering'},
                }
            ],
        }
    )

    actual = LatticeModel(manifest, {})(coords)
    expected = morton_sort_coords(coords)
    mx.eval(
        actual.coords,
        actual.order,
        actual.inverse_rows,
        expected.coords,
        expected.order,
        expected.inverse_rows,
    )

    assert actual.coords.tolist() == expected.coords.tolist()
    assert actual.order.tolist() == expected.order.tolist()
    assert actual.inverse_rows.tolist() == expected.inverse_rows.tolist()


def test_lattice_model_supports_typed_occupancy_outputs() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
        dtype=mx.int32,
    )
    # The generic signature for occupancy_expand accepts dense coord and
    # occupancy arrays, so build this graph explicitly with the public results
    # as a typed-output validation instead of wiring struct fields through IR.
    occupancy_manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [{'name': 'coords', 'type': 'dense_tensor'}],
            'outputs': [{'name': 'occupancy', 'type': 'sparse_occupancy'}],
            'nodes': [
                {
                    'id': 'downsample',
                    'op': 'ops.occupancy_downsample',
                    'inputs': {'coords': 'coords'},
                    'outputs': {'output': 'occupancy'},
                }
            ],
        }
    )
    actual_occupancy = LatticeModel(occupancy_manifest, {})(coords)
    expected_occupancy = occupancy_downsample(coords)
    expansion_manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [
                {'name': 'coords', 'type': 'dense_tensor'},
                {'name': 'occupancy', 'type': 'dense_tensor'},
            ],
            'outputs': [
                {'name': 'expanded', 'type': 'occupancy_expansion'}
            ],
            'nodes': [
                {
                    'id': 'expand',
                    'op': 'ops.occupancy_expand',
                    'inputs': {
                        'coords': 'coords',
                        'occupancy': 'occupancy',
                    },
                    'outputs': {'output': 'expanded'},
                }
            ],
        }
    )
    actual_expansion = LatticeModel(expansion_manifest, {})(
        expected_occupancy.coords,
        expected_occupancy.occupancy,
    )
    expected_expansion = occupancy_expand(
        expected_occupancy.coords,
        expected_occupancy.occupancy,
    )
    mx.eval(
        actual_occupancy.coords,
        actual_occupancy.active_rows,
        actual_occupancy.occupancy,
        expected_occupancy.coords,
        expected_occupancy.active_rows,
        expected_occupancy.occupancy,
        actual_expansion.coords,
        actual_expansion.parent_rows,
        actual_expansion.child_indices,
        expected_expansion.coords,
        expected_expansion.parent_rows,
        expected_expansion.child_indices,
    )

    occupancy_rows = _active_rows(actual_occupancy)
    assert occupancy_rows == _active_rows(expected_occupancy)
    _assert_active_array_equal(
        actual_occupancy.coords,
        expected_occupancy.coords,
        occupancy_rows,
    )
    assert (
        actual_occupancy.active_rows.tolist()
        == expected_occupancy.active_rows.tolist()
    )
    _assert_active_array_equal(
        actual_occupancy.occupancy,
        expected_occupancy.occupancy,
        occupancy_rows,
    )
    expansion_rows = _active_rows(actual_expansion)
    assert expansion_rows == _active_rows(expected_expansion)
    _assert_active_array_equal(
        actual_expansion.coords,
        expected_expansion.coords,
        expansion_rows,
    )
    _assert_active_array_equal(
        actual_expansion.parent_rows,
        expected_expansion.parent_rows,
        expansion_rows,
    )
    _assert_active_array_equal(
        actual_expansion.child_indices,
        expected_expansion.child_indices,
        expansion_rows,
    )


def test_lattice_graph_builder_projects_structured_fields() -> None:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
        dtype=mx.int32,
    )
    builder = LatticeGraphBuilder(
        inputs=cast('dict[str, IRValueType]', {'coords': 'dense_tensor'})
    )
    occupancy = builder.add_op(
        'downsample',
        'ops.occupancy_downsample',
        inputs={'coords': 'coords'},
    )
    downsampled_coords = builder.field(occupancy, 'coords')
    occupancy_mask = builder.field(occupancy, 'occupancy')
    expanded = builder.add_op(
        'expand',
        'ops.occupancy_expand',
        inputs={
            'coords': downsampled_coords,
            'occupancy': occupancy_mask,
        },
    )
    child_indices = builder.field(expanded, 'child_indices')
    artifact = build_lattice_graph_artifact(
        builder,
        outputs={
            occupancy: builder.output(occupancy, name='occupancy'),
            expanded: builder.output(expanded, name='expanded'),
            child_indices: builder.output(
                child_indices, name='child_indices'
            ),
        },
    )

    actual_occupancy, actual_expansion, actual_child_indices = LatticeModel(
        artifact.manifest,
        artifact.weights,
    )(coords)
    expected_occupancy = occupancy_downsample(coords)
    expected_expansion = occupancy_expand(
        expected_occupancy.coords,
        expected_occupancy.occupancy,
    )
    mx.eval(
        actual_occupancy.coords,
        actual_occupancy.occupancy,
        actual_expansion.coords,
        actual_expansion.child_indices,
        actual_child_indices,
        expected_occupancy.coords,
        expected_occupancy.occupancy,
        expected_expansion.coords,
        expected_expansion.child_indices,
    )

    assert [item.type for item in artifact.manifest.outputs] == [
        'sparse_occupancy',
        'occupancy_expansion',
        'dense_tensor',
    ]
    occupancy_rows = _active_rows(actual_occupancy)
    assert occupancy_rows == _active_rows(expected_occupancy)
    _assert_active_array_equal(
        actual_occupancy.coords,
        expected_occupancy.coords,
        occupancy_rows,
    )
    _assert_active_array_equal(
        actual_occupancy.occupancy,
        expected_occupancy.occupancy,
        occupancy_rows,
    )
    expansion_rows = _active_rows(actual_expansion)
    assert expansion_rows == _active_rows(expected_expansion)
    _assert_active_array_equal(
        actual_expansion.coords,
        expected_expansion.coords,
        expansion_rows,
    )
    _assert_active_array_equal(
        actual_child_indices,
        actual_expansion.child_indices,
        expansion_rows,
    )


def test_lattice_field_projection_uses_runtime_value_type_contract() -> (
    None
):
    assert value_type_fields('sparse_occupancy') == {
        'coords': 'dense_tensor',
        'active_rows': 'dense_tensor',
        'occupancy': 'dense_tensor',
    }
    assert value_type_fields('occupancy_expansion') == {
        'coords': 'dense_tensor',
        'active_rows': 'dense_tensor',
        'parent_rows': 'dense_tensor',
        'child_indices': 'dense_tensor',
    }


def test_lattice_graph_builder_rejects_unsupported_structured_field() -> (
    None
):
    builder = LatticeGraphBuilder(
        inputs=cast('dict[str, IRValueType]', {'coords': 'dense_tensor'})
    )
    occupancy = builder.add_op(
        'downsample',
        'ops.occupancy_downsample',
        inputs={'coords': 'coords'},
    )

    with pytest.raises(ValueError, match='not supported'):
        builder.field(occupancy, 'missing')


def test_lattice_module_artifact_roundtrips_sequential_graph(
    tmp_path,
) -> None:
    x = _input()
    conv = lnn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
    linear_module = lnn.Linear(1, 2)
    model = mxnn.Sequential(
        conv,
        lnn.ReLU(),
        linear_module,
        lnn.GlobalAvgPool(),
    )
    conv.weight = _weights()['conv.weight']
    linear_module.weight = _weights()['linear.weight']
    linear_module.bias = _weights()['linear.bias']

    artifact = build_lattice_module_artifact(
        model,
        output_name='logits',
        output_type='dense_tensor',
    )
    save_lattice_module(
        tmp_path,
        model,
        output_name='logits',
        output_type='dense_tensor',
    )

    in_memory = LatticeModel(artifact.manifest, artifact.weights)(x)
    loaded = load_lattice_model(tmp_path)(x)
    expected_sparse = linear(
        relu(conv3d(x, conv.weight, kernel_size=(3, 1, 1))),
        linear_module.weight,
        linear_module.bias,
    )
    expected = global_avg_pool(
        SparseTensor(
            expected_sparse.coords,
            expected_sparse.feats,
            expected_sparse.stride,
            coord_key=expected_sparse.coord_key,
            coord_manager=expected_sparse.coord_manager,
            batch_counts=(3,),
            active_rows=expected_sparse.active_rows,
        )
    )
    mx.eval(in_memory, loaded, expected)

    assert_nested_close(in_memory.tolist(), expected.tolist())
    assert_nested_close(loaded.tolist(), expected.tolist())


def test_lattice_module_artifact_infers_output_type() -> None:
    artifact = build_lattice_module_artifact(
        mxnn.Sequential(lnn.Linear(1, 1), lnn.GlobalAvgPool()),
        output_name='pooled',
    )

    assert artifact.manifest.outputs[0].name == 'pooled'
    assert artifact.manifest.outputs[0].type == 'dense_tensor'


def test_lattice_module_artifact_rejects_incompatible_input_type() -> None:
    with pytest.raises(ValueError, match=r"expects 'sparse_tensor'"):
        build_lattice_module_artifact(
            lnn.Linear(1, 1), input_type='dense_tensor'
        )


@pytest.mark.parametrize('bits', [4, 8])
def test_lattice_module_artifact_preserves_quantized_linear_storage(
    bits: int,
    tmp_path,
) -> None:
    x = _input()
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    quantized = dense.to_quantized(bits=bits)

    artifact = build_lattice_module_artifact(quantized)
    save_lattice_module(tmp_path, quantized)
    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    loaded = load_lattice_model(tmp_path)(x)
    expected = quantized(x)
    mx.eval(actual.feats, loaded.feats, expected.feats)

    assert artifact.manifest.nodes[0].op == 'feature.quantized_linear'
    prefix = artifact.manifest.nodes[0].parameters['weight']
    assert artifact.weights[f'{prefix}.weight'].dtype == mx.uint32
    attrs = cast('list[int]', artifact.weights[f'{prefix}.attrs'].tolist())
    assert attrs[2] == bits
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())
    assert_nested_close(loaded.feats.tolist(), expected.feats.tolist())


@pytest.mark.parametrize('bits', [4, 8])
def test_explicit_graph_builder_call_supports_quantized_parameters(
    bits: int,
) -> None:
    x = _input()
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    quantized = dense.to_quantized(bits=bits)
    builder = LatticeGraphBuilder()
    out = builder.call(
        'feature.quantized_linear',
        input='input',
        weight=quantized._quantized_weight(),
        bias=quantized.bias,
    )
    artifact = build_lattice_graph_artifact(builder, outputs=[out])
    weight = artifact.manifest.nodes[0].parameters['weight']

    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    expected = quantized(x)
    mx.eval(actual.feats, expected.feats)

    assert artifact.weights[f'{weight}.weight'].dtype == mx.uint32
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_explicit_graph_builder_call_supports_dense_parameters() -> None:
    x = _input()
    builder = LatticeGraphBuilder()
    out = builder.call(
        'feature.linear',
        input='input',
        weight=_weights()['linear.weight'],
        bias=_weights()['linear.bias'],
    )
    artifact = build_lattice_graph_artifact(builder, outputs=[out])

    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    expected = linear(
        x, _weights()['linear.weight'], _weights()['linear.bias']
    )
    mx.eval(actual.feats, expected.feats)

    assert set(artifact.manifest.nodes[0].parameters) == {'weight', 'bias'}
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_explicit_graph_builder_call_supports_generic_dense_weight_ops() -> (
    None
):
    x = _input()
    builder = LatticeGraphBuilder()
    out = builder.call(
        'ops.linear',
        x='input',
        weight=_weights()['linear.weight'],
        bias=_weights()['linear.bias'],
    )
    artifact = build_lattice_graph_artifact(builder, outputs=[out])

    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    expected = linear(
        x, _weights()['linear.weight'], _weights()['linear.bias']
    )
    mx.eval(actual.feats, expected.feats)

    assert artifact.manifest.nodes[0].op == 'ops.linear'
    assert set(artifact.manifest.nodes[0].parameters) == {'weight', 'bias'}
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_explicit_graph_builder_call_supports_generic_quantized_weight_ops() -> (
    None
):
    x = _input()
    dense = lnn.Linear(1, 2)
    dense.weight = _weights()['linear.weight']
    dense.bias = _weights()['linear.bias']
    quantized = dense.to_quantized(bits=4)
    builder = LatticeGraphBuilder()
    out = builder.call(
        'ops.linear',
        x='input',
        weight=quantized._quantized_weight(),
        bias=quantized.bias,
    )
    artifact = build_lattice_graph_artifact(builder, outputs=[out])
    weight = artifact.manifest.nodes[0].parameters['weight']

    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    expected = quantized(x)
    mx.eval(actual.feats, expected.feats)

    assert artifact.manifest.nodes[0].op == 'ops.linear'
    assert artifact.weights[f'{weight}.weight'].dtype == mx.uint32
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_custom_module_artifact_protocol_supports_skip_connection() -> None:
    class Residual(mxnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = lnn.Linear(1, 1, bias=False)
            self.proj.weight = mx.array([[2.0]], dtype=mx.float32)

        def __call__(self, x: SparseTensor) -> SparseTensor:
            return self.proj(x)

        def build_lattice_graph(
            self,
            builder: LatticeGraphBuilder,
            input_name: str,
        ) -> str:
            projected = builder.add_module('proj', self.proj, input_name)
            return builder.add_op(
                'skip',
                'ops.sparse_add',
                inputs={'lhs': input_name, 'rhs': projected},
                attributes={'join': 'outer'},
            )

    x = _input()
    artifact = build_lattice_module_artifact(Residual())
    actual = LatticeModel(artifact.manifest, artifact.weights)(x)
    expected = SparseTensor(
        x.coords,
        x.feats * 3,
        x.stride,
        coord_key=x.coord_key,
        coord_manager=x.coord_manager,
        batch_counts=x.batch_counts,
        active_rows=x.active_rows,
    )
    mx.eval(actual.feats, expected.feats)

    assert [node.op for node in artifact.manifest.nodes] == [
        'feature.linear',
        'ops.sparse_add',
    ]
    assert_nested_close(actual.feats.tolist(), expected.feats.tolist())


def test_explicit_graph_artifact_supports_multiple_inputs_and_outputs(
    tmp_path,
) -> None:
    builder = LatticeGraphBuilder(
        inputs=cast(
            'dict[str, IRValueType]',
            {
                'points': 'dense_tensor',
                'feats': 'dense_tensor',
                'batches': 'dense_tensor',
            },
        )
    )
    voxels = builder.add_op(
        'voxelize',
        'ops.voxelize',
        inputs={'points': 'points', 'feats': 'feats'},
        attributes={
            'batch_indices': 'batches',
            'reduction': 'mean',
            'voxel_size': 1.0,
        },
    )
    sampled = builder.add_op(
        'devoxelize',
        'ops.devoxelize',
        inputs={'points': 'points', 'voxels': voxels},
        attributes={
            'batch_indices': 'batches',
            'interpolation': 'nearest',
            'voxel_size': 1.0,
        },
    )
    outputs = {
        voxels: GraphOutput(voxels, 'sparse_tensor', 'voxels'),
        sampled: GraphOutput(sampled, 'dense_tensor', 'sampled'),
    }
    artifact = build_lattice_graph_artifact(builder, outputs=outputs)
    save_lattice_graph(tmp_path, builder, outputs=outputs)
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)
    batches = mx.array([0, 0, 0], dtype=mx.int32)

    actual_voxels, actual_sampled = LatticeModel(
        artifact.manifest,
        artifact.weights,
    )(points, feats, batches)
    loaded_voxels, loaded_sampled = load_lattice_model(tmp_path)(
        points,
        feats,
        batches,
    )
    mx.eval(
        actual_voxels.feats,
        actual_sampled,
        loaded_voxels.feats,
        loaded_sampled,
    )

    assert [item.name for item in artifact.manifest.outputs] == [
        'voxels',
        'sampled',
    ]
    assert actual_voxels.coords.tolist() == loaded_voxels.coords.tolist()
    assert_nested_close(
        actual_voxels.feats.tolist(),
        loaded_voxels.feats.tolist(),
    )
    assert_nested_close(actual_sampled.tolist(), loaded_sampled.tolist())


def test_explicit_graph_builder_call_uses_registered_argument_contracts() -> (
    None
):
    builder = LatticeGraphBuilder(
        inputs=cast(
            'dict[str, IRValueType]',
            {
                'points': 'dense_tensor',
                'feats': 'dense_tensor',
                'batches': 'dense_tensor',
            },
        )
    )
    quantization = builder.call(
        'ops.sparse_quantize',
        points='points',
        batch_indices='batches',
        voxel_size=1.0,
    )
    voxels = builder.call(
        'ops.voxelize_with_quantization',
        quantization=quantization,
        feats='feats',
        reduction='mean',
    )
    sampled = builder.call(
        'ops.devoxelize',
        points='points',
        voxels=voxels,
        batch_indices='batches',
        interpolation='nearest',
    )
    artifact = build_lattice_graph_artifact(
        builder,
        outputs={
            voxels: builder.output(voxels, name='voxels'),
            sampled: builder.output(sampled, name='sampled'),
        },
    )
    points = mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )
    feats = mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)
    batches = mx.array([0, 0, 0], dtype=mx.int32)

    actual_voxels, actual_sampled = LatticeModel(
        artifact.manifest,
        artifact.weights,
    )(points, feats, batches)
    expected_quantization = lops.sparse_quantize(
        points,
        batch_indices=batches,
        voxel_size=1.0,
    )
    expected_voxels = lops.voxelize_with_quantization(
        expected_quantization,
        feats,
        reduction='mean',
    )
    expected_sampled = lops.devoxelize(
        points,
        expected_voxels,
        batch_indices=batches,
        interpolation='nearest',
    )
    mx.eval(
        actual_voxels.coords,
        actual_voxels.feats,
        actual_sampled,
        expected_voxels.coords,
        expected_voxels.feats,
        expected_sampled,
    )

    assert actual_voxels.coords.tolist() == expected_voxels.coords.tolist()
    assert_nested_close(
        actual_voxels.feats.tolist(), expected_voxels.feats.tolist()
    )
    assert_nested_close(actual_sampled.tolist(), expected_sampled.tolist())


def test_explicit_graph_builder_rejects_invalid_registered_call() -> None:
    builder = LatticeGraphBuilder()

    with pytest.raises(ValueError, match='missing required keys'):
        builder.call('feature.linear', input='input')

    with pytest.raises(ValueError, match='unsupported arguments'):
        builder.call('feature.relu', input='input', unknown='value')

    with pytest.raises(ValueError, match='requires QuantizedWeight'):
        builder.call(
            'feature.quantized_linear',
            input='input',
            weight=_weights()['linear.weight'],
        )

    dense_builder = LatticeGraphBuilder(
        inputs=cast('dict[str, IRValueType]', {'dense': 'dense_tensor'})
    )
    with pytest.raises(ValueError, match=r"expects 'sparse_tensor'"):
        dense_builder.call('feature.relu', input='dense')


def test_explicit_graph_artifact_infers_output_types() -> None:
    builder = LatticeGraphBuilder(
        inputs=cast(
            'dict[str, IRValueType]',
            {
                'points': 'dense_tensor',
                'feats': 'dense_tensor',
            },
        )
    )
    voxels = builder.add_op(
        'voxelize',
        'ops.voxelize',
        inputs={'points': 'points', 'feats': 'feats'},
    )
    offsets = builder.add_op(
        'offsets',
        'ops.kernel_offsets',
        inputs={},
        attributes={'kernel_size': [1, 1, 1]},
    )
    rows = builder.add_op(
        'topk',
        'ops.topk_rows',
        inputs={'x': voxels},
        attributes={'counts': [1]},
    )

    artifact = build_lattice_graph_artifact(
        builder, outputs=[voxels, rows, offsets]
    )
    aliased = build_lattice_graph_artifact(
        builder,
        outputs={
            voxels: builder.output(voxels, name='voxels'),
            rows: builder.output(rows, name='rows'),
            offsets: builder.output(offsets, name='offsets'),
        },
    )

    assert [item.type for item in artifact.manifest.outputs] == [
        'sparse_tensor',
        'dense_tensor',
        'any',
    ]
    assert [item.name for item in aliased.manifest.outputs] == [
        'voxels',
        'rows',
        'offsets',
    ]


def test_explicit_graph_builder_tracks_declared_output_ports_independently() -> (
    None
):
    from mlx_lattice.artifact.registry import lattice_op

    @lattice_op(
        '__test.multi_output',
        function=lambda: None,
        inputs={},
        outputs={'dense', 'sparse'},
        output='dense',
        output_types=cast(
            'dict[str, IRValueType]',
            {'dense': 'dense_tensor', 'sparse': 'sparse_tensor'},
        ),
    )
    def _handler(context, node):
        raise AssertionError(
            'builder type inference should not execute ops'
        )

    builder = LatticeGraphBuilder()

    primary = builder.add_op(
        'custom',
        '__test.multi_output',
        inputs={},
        outputs={'dense': 'dense_value', 'sparse': 'sparse_value'},
    )

    assert primary == 'dense_value'
    assert builder.value_types['dense_value'] == 'dense_tensor'
    assert builder.value_types['sparse_value'] == 'sparse_tensor'


def test_lattice_model_rejects_runtime_output_port_mismatch() -> None:
    from mlx_lattice.artifact.registry import lattice_op

    @lattice_op(
        '__test.missing_runtime_output',
        function=lambda: None,
        inputs={},
        output_types=cast(
            'dict[str, IRValueType]', {'output': 'dense_tensor'}
        ),
    )
    def _handler(context, node):
        return {}

    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [],
            'outputs': [{'name': 'output', 'type': 'dense_tensor'}],
            'nodes': [
                {
                    'id': 'bad',
                    'op': '__test.missing_runtime_output',
                    'outputs': {'output': 'output'},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match='artifact_outputs missing'):
        LatticeModel(manifest, {})()


def test_lattice_model_rejects_runtime_output_type_mismatch() -> None:
    from mlx_lattice.artifact.registry import lattice_op

    @lattice_op(
        '__test.wrong_runtime_output_type',
        function=lambda: None,
        inputs={},
        output_types=cast(
            'dict[str, IRValueType]', {'output': 'dense_tensor'}
        ),
    )
    def _handler(context, node):
        return {'output': _input()}

    manifest = manifest_from_dict(
        {
            'schema_version': '0.1',
            'inputs': [],
            'outputs': [{'name': 'output', 'type': 'dense_tensor'}],
            'nodes': [
                {
                    'id': 'bad',
                    'op': '__test.wrong_runtime_output_type',
                    'outputs': {'output': 'output'},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match='must be a dense tensor'):
        LatticeModel(manifest, {})()
