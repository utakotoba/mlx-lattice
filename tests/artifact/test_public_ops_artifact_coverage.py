from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Protocol, cast

import pytest
from lattice_contract import IRValueType

from mlx_lattice import SparseTensor
from mlx_lattice import ops as lops
from mlx_lattice.artifact import (
    LatticeGraphBuilder,
    LatticeModel,
    build_lattice_graph_artifact,
)
from tests.support import assert_nested_close, mx

pytestmark = [pytest.mark.usefixtures('selected_backend')]


@dataclass(frozen=True, slots=True)
class PublicOpCase:
    inputs: dict[str, tuple[IRValueType, object]]
    arguments: dict[str, object]
    outputs: tuple[str, ...] = ('output',)
    expected: Callable[[], object] | None = None
    value_type: IRValueType | None = None
    fields: tuple[str, ...] = ()
    tolerates_dtype_policy: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class _CapacityValue(Protocol):
    capacity: int


class _NeighborEdgesValue(Protocol):
    query_rows: mx.array
    source_rows: mx.array
    neighbor_ids: mx.array


class _ActiveCoordinateValue(Protocol):
    coords: mx.array
    active_rows: mx.array
    capacity: int


def test_public_ops_artifact_cases_cover_every_public_function() -> None:
    public = {
        name
        for name in lops.__all__
        if inspect.isfunction(getattr(lops, name))
    }

    assert set(_PUBLIC_OP_CASES) == public


def _resolve_expected_argument(
    value: object,
    inputs: dict[str, tuple[IRValueType, object]],
) -> object:
    if isinstance(value, str) and value in inputs:
        return inputs[value][1]
    if isinstance(value, tuple) and all(
        isinstance(item, str) for item in value
    ):
        names = cast('tuple[str, ...]', value)
        return tuple(inputs[item][1] for item in names)
    return value


def _assert_equivalent(
    actual: object,
    expected: object,
    *,
    fields: tuple[str, ...] = (),
) -> None:
    if fields and _is_active_coordinate_struct(actual, expected):
        _assert_active_coordinate_struct(actual, expected)
        return
    if fields:
        for field in fields:
            _assert_equivalent(
                getattr(actual, field), getattr(expected, field)
            )
        return
    if isinstance(actual, SparseTensor) and isinstance(
        expected, SparseTensor
    ):
        actual_rows = _active_count(actual)
        expected_rows = _active_count(expected)
        mx.eval(
            actual.coords, actual.feats, expected.coords, expected.feats
        )
        assert actual.stride == expected.stride
        assert actual_rows == expected_rows
        assert (
            actual.coords[:actual_rows].tolist()
            == expected.coords[:expected_rows].tolist()
        )
        assert_nested_close(
            actual.feats[:actual_rows].tolist(),
            expected.feats[:expected_rows].tolist(),
        )
        return
    if _is_neighbor_edges(actual, expected):
        _assert_neighbor_edges(actual, expected)
        return
    if _is_active_coordinate_struct(actual, expected):
        _assert_active_coordinate_struct(actual, expected)
        return
    if is_dataclass(actual) and is_dataclass(expected):
        for item in dataclass_fields(actual):
            _assert_equivalent(
                getattr(actual, item.name),
                getattr(expected, item.name),
            )
        return
    if isinstance(actual, mx.array) and isinstance(expected, mx.array):
        mx.eval(actual, expected)
        assert actual.shape == expected.shape
        if actual.dtype == mx.bool_ or actual.dtype in (
            mx.int8,
            mx.int16,
            mx.int32,
            mx.int64,
            mx.uint32,
        ):
            assert actual.tolist() == expected.tolist()
        else:
            assert_nested_close(actual.tolist(), expected.tolist())
        return
    if isinstance(actual, bytes) and isinstance(expected, bytes):
        assert actual == expected
        return
    assert actual == expected


def _active_count(value: object) -> int:
    active_rows = getattr(value, 'active_rows', None)
    if isinstance(active_rows, mx.array):
        return int(active_rows.tolist()[0])
    return int(cast('_CapacityValue', value).capacity)


def _is_neighbor_edges(actual: object, expected: object) -> bool:
    return all(
        hasattr(value, field)
        for value in (actual, expected)
        for field in ('query_rows', 'source_rows', 'neighbor_ids')
    )


def _assert_neighbor_edges(actual: object, expected: object) -> None:
    actual_values = _edge_rows(actual)
    expected_values = _edge_rows(expected)
    assert sorted(actual_values) == sorted(expected_values)


def _edge_rows(value: object) -> list[tuple[int, int, int]]:
    edges = cast('_NeighborEdgesValue', value)
    query_rows = edges.query_rows
    source_rows = edges.source_rows
    neighbor_ids = edges.neighbor_ids
    mx.eval(query_rows, source_rows, neighbor_ids)
    return [
        (int(query), int(source), int(neighbor))
        for query, source, neighbor in zip(
            query_rows.tolist(),
            source_rows.tolist(),
            neighbor_ids.tolist(),
            strict=True,
        )
    ]


def _is_active_coordinate_struct(actual: object, expected: object) -> bool:
    return all(
        hasattr(value, field)
        for value in (actual, expected)
        for field in ('coords', 'active_rows')
    )


def _assert_active_coordinate_struct(
    actual: object, expected: object
) -> None:
    actual_value = cast('_ActiveCoordinateValue', actual)
    expected_value = cast('_ActiveCoordinateValue', expected)
    actual_rows = _active_count(actual_value)
    expected_rows = _active_count(expected_value)
    assert actual_rows == expected_rows
    _assert_equivalent(
        actual_value.coords[:actual_rows],
        expected_value.coords[:expected_rows],
    )
    if is_dataclass(actual) and is_dataclass(expected):
        for item in dataclass_fields(actual):
            if item.name in ('coords', 'active_rows'):
                continue
            value = getattr(actual, item.name)
            other = getattr(expected, item.name)
            if isinstance(value, mx.array) and value.shape[:1] == (
                actual_value.capacity,
            ):
                _assert_equivalent(
                    value[:actual_rows], other[:expected_rows]
                )
            else:
                _assert_equivalent(value, other)


def _sparse() -> SparseTensor:
    return SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        batch_counts=(3,),
    )


def _sparse_alt() -> SparseTensor:
    return SparseTensor(
        mx.array(
            [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[0.5], [1.5], [2.5]], dtype=mx.float32),
        batch_counts=(3,),
    )


def _coords() -> mx.array:
    return mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )


def _target_coords() -> mx.array:
    return mx.array(
        [[0, 1, 0, 0], [0, 3, 0, 0]],
        dtype=mx.int32,
    )


def _points() -> mx.array:
    return mx.array(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=mx.float32,
    )


def _point_feats() -> mx.array:
    return mx.array([[1.0], [3.0], [5.0]], dtype=mx.float32)


def _prob() -> mx.array:
    return mx.array([[0.2, 0.8], [0.7, 0.3]], dtype=mx.float32)


def _symbols() -> mx.array:
    return mx.array([1, 0], dtype=mx.int32)


def _stream() -> bytes:
    return lops.range_encode_from_prob(_prob(), _symbols())


def _case(
    *args: tuple[str, IRValueType, object],
    arguments: dict[str, object],
    expected: Callable[[], object] | None = None,
    value_type: IRValueType | None = None,
    fields: tuple[str, ...] = (),
) -> PublicOpCase:
    return PublicOpCase(
        inputs={
            name: (value_type_, value) for name, value_type_, value in args
        },
        arguments=arguments,
        expected=expected,
        value_type=value_type,
        fields=fields,
    )


_X = _sparse()
_Y = _sparse_alt()
_COORDS = _coords()
_TARGET = _target_coords()
_POINTS = _points()
_POINT_FEATS = _point_feats()
_BATCHES = mx.array([0, 0, 0], dtype=mx.int32)
_ACTIVE = mx.array([3], dtype=mx.int32)
_WEIGHT = mx.array([[2.0]], dtype=mx.float32)
_BIAS = mx.array([1.0], dtype=mx.float32)
_CONV_WEIGHT = mx.array([1.0], dtype=mx.float32).reshape(1, 1, 1, 1, 1)
_MASK = mx.array([False, True, True], dtype=mx.bool_)
_ROWS = mx.array([0, 2], dtype=mx.int32)
_REPLACE_FEATS = mx.array([[2.0], [4.0], [6.0]], dtype=mx.float32)
_QUANTIZATION = lops.sparse_quantize(_POINTS, batch_indices=_BATCHES)
_VOXELS = lops.voxelize(_POINTS, _POINT_FEATS, batch_indices=_BATCHES)
_POINT_MAP = lops.build_point_voxel_map(
    _POINTS,
    _VOXELS.coords,
    _VOXELS.active_rows,
    batch_indices=_BATCHES,
    interpolation='nearest',
)
_OCCUPANCY = lops.occupancy_downsample(_COORDS)
_ORDERING = lops.morton_sort_coords(_COORDS)
_KNN = lops.knn_relation(_X, k=1)
_CDF = lops.normalized_cdf(_prob())
_STREAM = _stream()

_PUBLIC_OP_CASES = {
    'align_sparse': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs', 'join': 'outer'},
        fields=('coords', 'active_rows', 'lhs_rows', 'rhs_rows'),
    ),
    'avg_pool3d': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'batch_norm': _case(
        ('x', 'sparse_tensor', _X),
        arguments={
            'x': 'x',
            'weight': mx.array([1.0]),
            'bias': mx.array([0.0]),
        },
    ),
    'build_generative_relation': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
    ),
    'build_kernel_relation': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords', 'kernel_size': 1},
    ),
    'build_knn_relation': _case(
        ('source_coords', 'dense_tensor', _COORDS),
        arguments={'source_coords': 'source_coords', 'k': 1},
    ),
    'build_point_voxel_map': _case(
        ('points', 'dense_tensor', _POINTS),
        ('voxel_coords', 'dense_tensor', _VOXELS.coords),
        ('voxel_active_rows', 'dense_tensor', _VOXELS.active_rows),
        ('batch_indices', 'dense_tensor', _BATCHES),
        arguments={
            'points': 'points',
            'voxel_coords': 'voxel_coords',
            'voxel_active_rows': 'voxel_active_rows',
            'batch_indices': 'batch_indices',
            'interpolation': 'nearest',
        },
        fields=('rows', 'weights'),
    ),
    'build_radius_relation': _case(
        ('source_coords', 'dense_tensor', _COORDS),
        arguments={
            'source_coords': 'source_coords',
            'radius': 1.1,
            'max_neighbors': 3,
        },
        fields=('counts',),
    ),
    'build_sparse_alignment': _case(
        ('lhs_coords', 'dense_tensor', _X.coords),
        ('lhs_active_rows', 'dense_tensor', _X.active_rows),
        ('rhs_coords', 'dense_tensor', _Y.coords),
        ('rhs_active_rows', 'dense_tensor', _Y.active_rows),
        arguments={
            'lhs_coords': 'lhs_coords',
            'lhs_active_rows': 'lhs_active_rows',
            'rhs_coords': 'rhs_coords',
            'rhs_active_rows': 'rhs_active_rows',
            'join': 'outer',
        },
        fields=('coords', 'active_rows', 'lhs_rows', 'rhs_rows'),
    ),
    'build_submanifold_kernel_relation': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords', 'kernel_size': 1},
    ),
    'build_target_kernel_relation': _case(
        ('coords', 'dense_tensor', _COORDS),
        ('target_coords', 'dense_tensor', _TARGET),
        arguments={
            'coords': 'coords',
            'target_coords': 'target_coords',
            'kernel_size': 1,
        },
        fields=('counts',),
    ),
    'build_transposed_kernel_relation': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords', 'kernel_size': 1, 'stride': 1},
    ),
    'cat': _case(
        ('x', 'sparse_tensor', _X),
        ('y', 'sparse_tensor', _X.replace(feats=_X.feats + 1)),
        arguments={'tensors': ('x', 'y')},
    ),
    'child_coords_from_indices': _case(
        ('parent_coords', 'dense_tensor', _OCCUPANCY.coords),
        ('child_indices', 'dense_tensor', _OCCUPANCY.occupancy),
        arguments={
            'parent_coords': 'parent_coords',
            'child_indices': 'child_indices',
        },
    ),
    'contains_coords': _case(
        ('coords', 'dense_tensor', _COORDS),
        ('queries', 'dense_tensor', _TARGET),
        arguments={'coords': 'coords', 'queries': 'queries'},
    ),
    'conv3d': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'weight': _CONV_WEIGHT, 'kernel_size': 1},
    ),
    'conv_transpose3d': _case(
        ('x', 'sparse_tensor', _X),
        arguments={
            'x': 'x',
            'weight': _CONV_WEIGHT,
            'kernel_size': 1,
            'stride': 1,
        },
    ),
    'crop': _case(
        ('x', 'sparse_tensor', _X),
        arguments={
            'x': 'x',
            'min_coord': [1, 0, 0],
            'max_coord': [2, 0, 0],
        },
    ),
    'devoxelize': _case(
        ('points', 'dense_tensor', _POINTS),
        ('voxels', 'sparse_tensor', _VOXELS),
        ('batches', 'dense_tensor', _BATCHES),
        arguments={
            'points': 'points',
            'voxels': 'voxels',
            'batch_indices': 'batches',
            'interpolation': 'nearest',
        },
    ),
    'downsample_coords': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
        fields=('coords', 'active_rows'),
    ),
    'dropout': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'training': False},
    ),
    'gather_aligned_features': _case(
        ('x', 'sparse_tensor', _X),
        ('rows', 'dense_tensor', mx.array([0, -1, 2], dtype=mx.int32)),
        arguments={'x': 'x', 'rows': 'rows'},
    ),
    'gather_neighbor_features': _case(
        ('source', 'sparse_tensor', _X),
        ('relation', 'relation', _KNN),
        arguments={'source': 'source', 'relation': 'relation'},
    ),
    'gelu': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'generative_conv_transpose3d': _case(
        ('x', 'sparse_tensor', _X),
        arguments={
            'x': 'x',
            'weight': _CONV_WEIGHT,
            'kernel_size': 1,
            'stride': 1,
        },
    ),
    'generative_kernel_relation': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'kernel_size': 1, 'stride': 1},
    ),
    'global_avg_pool': _case(
        ('x', 'sparse_tensor', _X), arguments={'x': 'x'}
    ),
    'global_max_pool': _case(
        ('x', 'sparse_tensor', _X), arguments={'x': 'x'}
    ),
    'global_sum_pool': _case(
        ('x', 'sparse_tensor', _X), arguments={'x': 'x'}
    ),
    'interpolate_point_features': _case(
        ('voxel_feats', 'dense_tensor', _VOXELS.feats),
        ('point_voxel_map', 'point_voxel_map', _POINT_MAP),
        arguments={
            'voxel_feats': 'voxel_feats',
            'point_voxel_map': 'point_voxel_map',
        },
    ),
    'intersection_coords': _case(
        ('lhs', 'dense_tensor', _COORDS),
        ('rhs', 'dense_tensor', _TARGET),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
        fields=('coords', 'active_rows'),
    ),
    'inverse_map': _case(
        ('source', 'dense_tensor', _COORDS),
        ('target', 'dense_tensor', _TARGET),
        arguments={'source': 'source', 'target': 'target'},
    ),
    'kernel_offsets': _case(
        arguments={'kernel_size': 1},
        expected=lambda: lops.kernel_offsets(1),
        value_type='any',
    ),
    'kernel_relation': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'kernel_size': 1},
    ),
    'knn_relation': _case(
        ('source', 'sparse_tensor', _X),
        arguments={'source': 'source', 'k': 1},
    ),
    'layer_norm': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'weight': mx.array([1.0])},
    ),
    'leaky_relu': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'linear': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'weight': _WEIGHT, 'bias': _BIAS},
    ),
    'lookup_coords': _case(
        ('coords', 'dense_tensor', _COORDS),
        ('queries', 'dense_tensor', _TARGET),
        arguments={'coords': 'coords', 'queries': 'queries'},
    ),
    'max_pool3d': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'morton_codes': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
    ),
    'morton_order': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
    ),
    'morton_sort_coords': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
        fields=('coords', 'order', 'inverse_rows'),
    ),
    'normalized_cdf': _case(
        ('prob', 'dense_tensor', _prob()),
        arguments={'prob': 'prob'},
    ),
    'occupancy_downsample': _case(
        ('coords', 'dense_tensor', _COORDS),
        arguments={'coords': 'coords'},
        fields=('coords', 'active_rows', 'occupancy'),
    ),
    'occupancy_expand': _case(
        ('coords', 'dense_tensor', _OCCUPANCY.coords),
        ('occupancy', 'dense_tensor', _OCCUPANCY.occupancy),
        arguments={'coords': 'coords', 'occupancy': 'occupancy'},
        fields=('coords', 'active_rows', 'parent_rows', 'child_indices'),
    ),
    'pool3d': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'prune': _case(
        ('x', 'sparse_tensor', _X),
        ('rows', 'dense_tensor', _ROWS),
        arguments={'x': 'x', 'rows': 'rows'},
    ),
    'prune_mask': _case(
        ('x', 'sparse_tensor', _X),
        ('mask', 'dense_tensor', _MASK),
        arguments={'x': 'x', 'mask': 'mask'},
    ),
    'radius_relation': _case(
        ('source', 'sparse_tensor', _X),
        arguments={'source': 'source', 'radius': 1.1, 'max_neighbors': 3},
        fields=('counts',),
    ),
    'range_decode': _case(
        ('cdf', 'dense_tensor', _CDF),
        ('stream', 'bytes', lops.range_encode(_CDF, _symbols())),
        arguments={'cdf': 'cdf', 'stream': 'stream'},
    ),
    'range_decode_from_prob': _case(
        ('prob', 'dense_tensor', _prob()),
        ('stream', 'bytes', _STREAM),
        arguments={'prob': 'prob', 'stream': 'stream'},
    ),
    'range_encode': _case(
        ('cdf', 'dense_tensor', _CDF),
        ('symbols', 'dense_tensor', _symbols()),
        arguments={'cdf': 'cdf', 'symbols': 'symbols'},
    ),
    'range_encode_from_prob': _case(
        ('prob', 'dense_tensor', _prob()),
        ('symbols', 'dense_tensor', _symbols()),
        arguments={'prob': 'prob', 'symbols': 'symbols'},
    ),
    'rans_decode_from_prob': _case(
        ('prob', 'dense_tensor', _prob()),
        (
            'stream',
            'bytes',
            lops.rans_encode_from_prob(_prob(), _symbols()),
        ),
        arguments={'prob': 'prob', 'stream': 'stream'},
    ),
    'rans_encode_from_prob': _case(
        ('prob', 'dense_tensor', _prob()),
        ('symbols', 'dense_tensor', _symbols()),
        arguments={'prob': 'prob', 'symbols': 'symbols'},
    ),
    'relu': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'replace_feature': _case(
        ('x', 'sparse_tensor', _X),
        ('feats', 'dense_tensor', _REPLACE_FEATS),
        arguments={'x': 'x', 'feats': 'feats'},
    ),
    'rms_norm': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'weight': mx.array([1.0])},
    ),
    'sigmoid': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'silu': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'softplus': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'sparse_add': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
    ),
    'sparse_binary_op': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs', 'op': 'add'},
    ),
    'sparse_cat_aligned': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs', 'join': 'outer'},
    ),
    'sparse_collate': _case(
        ('coords0', 'dense_tensor', mx.array([[0, 0, 0]], dtype=mx.int32)),
        ('coords1', 'dense_tensor', mx.array([[1, 0, 0]], dtype=mx.int32)),
        ('feats0', 'dense_tensor', mx.array([[1.0]], dtype=mx.float32)),
        ('feats1', 'dense_tensor', mx.array([[2.0]], dtype=mx.float32)),
        arguments={
            'coords': ('coords0', 'coords1'),
            'feats': ('feats0', 'feats1'),
        },
    ),
    'sparse_maximum': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
    ),
    'sparse_minimum': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
    ),
    'sparse_mul': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
    ),
    'sparse_quantize': _case(
        ('points', 'dense_tensor', _POINTS),
        ('batch_indices', 'dense_tensor', _BATCHES),
        arguments={'points': 'points', 'batch_indices': 'batch_indices'},
        fields=('coords', 'active_rows', 'inverse_rows', 'counts'),
    ),
    'sparse_sub': _case(
        ('lhs', 'sparse_tensor', _X),
        ('rhs', 'sparse_tensor', _Y),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
    ),
    'subm_conv3d': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'weight': _CONV_WEIGHT, 'kernel_size': 1},
    ),
    'submanifold_kernel_relation': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'kernel_size': 1},
    ),
    'sum_pool3d': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'tanh': _case(('x', 'sparse_tensor', _X), arguments={'x': 'x'}),
    'target_kernel_relation': _case(
        ('x', 'sparse_tensor', _X),
        ('target', 'sparse_tensor', SparseTensor(_TARGET, mx.ones((2, 1)))),
        arguments={'x': 'x', 'target': 'target', 'kernel_size': 1},
    ),
    'topk_rows': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'counts': [2]},
    ),
    'transposed_kernel_relation': _case(
        ('x', 'sparse_tensor', _X),
        arguments={'x': 'x', 'kernel_size': 1, 'stride': 1},
    ),
    'union_coords': _case(
        ('lhs', 'dense_tensor', _COORDS),
        ('rhs', 'dense_tensor', _TARGET),
        arguments={'lhs': 'lhs', 'rhs': 'rhs'},
        fields=('coords', 'active_rows'),
    ),
    'voxelize': _case(
        ('points', 'dense_tensor', _POINTS),
        ('feats', 'dense_tensor', _POINT_FEATS),
        ('batch_indices', 'dense_tensor', _BATCHES),
        arguments={
            'points': 'points',
            'feats': 'feats',
            'batch_indices': 'batch_indices',
        },
    ),
    'voxelize_with_quantization': _case(
        ('quantization', 'quantization', _QUANTIZATION),
        ('feats', 'dense_tensor', _POINT_FEATS),
        arguments={'quantization': 'quantization', 'feats': 'feats'},
    ),
}


@pytest.mark.parametrize('name', sorted(_PUBLIC_OP_CASES))
def test_public_ops_roundtrip_through_explicit_artifact_graph(
    name: str,
) -> None:
    case = _PUBLIC_OP_CASES[name]
    builder = LatticeGraphBuilder(
        inputs={
            key: value_type for key, (value_type, _) in case.inputs.items()
        }
    )
    out = builder.call(
        f'ops.{name}',
        **cast('dict[str, Any]', case.arguments),
    )
    artifact = build_lattice_graph_artifact(
        builder,
        outputs={
            out: builder.output(
                out,
                name='output',
                value_type=case.value_type,
            )
        },
    )
    model = LatticeModel(artifact.manifest, artifact.weights)

    actual = model(*(value for _, value in case.inputs.values()))
    expected = (
        getattr(lops, name)(
            **{
                key: _resolve_expected_argument(value, case.inputs)
                for key, value in case.arguments.items()
            }
        )
        if case.expected is None
        else case.expected()
    )
    _assert_equivalent(actual, expected, fields=case.fields)
