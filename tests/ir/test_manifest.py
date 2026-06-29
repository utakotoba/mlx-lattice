from __future__ import annotations

import pytest

from mlx_lattice.export import LatticeModel
from mlx_lattice.ir import (
    ir_value_type,
    is_ir_value_type,
    iter_op_specs,
    manifest_from_dict,
    manifest_to_dict,
)


def _manifest() -> dict:
    return {
        'schema_version': '0.1',
        'producer': {'name': 'test'},
        'runtime': {'name': 'mlx-lattice', 'version': '>=0.2,<0.3'},
        'coordinate_order': ['batch', 'x', 'y', 'z'],
        'feature_layout': ['row', 'channel'],
        'weight_layout': 'mlx-lattice',
        'dtype_policy': 'preserve',
        'inputs': [{'name': 'input', 'type': 'sparse_tensor'}],
        'outputs': [{'name': 'output', 'type': 'sparse_tensor'}],
        'nodes': [
            {
                'id': 'relu',
                'op': 'feature.relu',
                'inputs': {'input': 'input'},
                'outputs': {'output': 'output'},
            }
        ],
    }


def test_manifest_roundtrip_preserves_semantic_contract() -> None:
    manifest = manifest_from_dict(_manifest())
    LatticeModel(manifest, {})

    raw = manifest_to_dict(manifest)

    assert raw['schema_version'] == '0.1'
    assert raw['coordinate_order'] == ['batch', 'x', 'y', 'z']
    assert raw['nodes'][0]['op'] == 'feature.relu'


def test_ir_op_specs_are_registry_backed() -> None:
    names = {spec.name for spec in iter_op_specs()}

    assert {
        'sparse.conv3d',
        'sparse.subm_conv3d',
        'sparse.add',
        'feature.linear',
        'feature.relu',
        'feature.gelu',
        'ops.voxelize',
        'ops.occupancy_downsample',
        'pool.global_avg',
    } <= names
    conv = next(
        spec for spec in iter_op_specs() if spec.name == 'sparse.conv3d'
    )
    assert {
        'kernel_size',
        'stride',
        'padding',
        'dilation',
    } <= conv.attributes
    assert 'coordinates' in conv.value_attributes
    assert conv.output_types == {'output': 'sparse_tensor'}
    global_avg = next(
        spec for spec in iter_op_specs() if spec.name == 'pool.global_avg'
    )
    assert global_avg.output_types == {'output': 'dense_tensor'}


def test_manifest_rejects_unknown_schema_version() -> None:
    raw = _manifest()
    raw['schema_version'] = '9.9'

    with pytest.raises(ValueError, match='unsupported lattice IR'):
        manifest_from_dict(raw)


def test_ir_value_type_helper_validates_schema_values() -> None:
    assert is_ir_value_type('sparse_tensor')
    assert ir_value_type('dense_tensor') == 'dense_tensor'
    assert not is_ir_value_type('not_a_value_type')

    with pytest.raises(ValueError, match='unsupported IR value type'):
        ir_value_type('not_a_value_type')


def test_manifest_rejects_unknown_graph_input_reference() -> None:
    raw = _manifest()
    raw['nodes'][0]['inputs']['input'] = 'missing'

    with pytest.raises(ValueError, match='unknown input'):
        manifest_from_dict(raw)


def test_manifest_rejects_duplicate_graph_values() -> None:
    raw = _manifest()
    raw['nodes'][0]['outputs']['output'] = 'input'

    with pytest.raises(ValueError, match='duplicate graph value'):
        manifest_from_dict(raw)
