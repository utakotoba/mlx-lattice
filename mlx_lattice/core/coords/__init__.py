from __future__ import annotations

from mlx_lattice.core.coords.builders import (
    build_generative_relation,
    build_kernel_relation,
    build_knn_relation,
    build_radius_relation,
    build_transposed_kernel_relation,
    kernel_offsets,
)
from mlx_lattice.core.coords.manager import (
    CoordinateManager,
    CoordinateMapKey,
)
from mlx_lattice.core.coords.quantization import (
    SparseQuantization,
    sparse_quantize,
)
from mlx_lattice.core.coords.set_ops import (
    CoordinateSet,
    contains_coords,
    downsample_coords,
    intersection_coords,
    inverse_map,
    lookup_coords,
    union_coords,
)
from mlx_lattice.core.coords.validation import validate_coords

__all__ = [
    'CoordinateManager',
    'CoordinateMapKey',
    'CoordinateSet',
    'SparseQuantization',
    'build_generative_relation',
    'build_kernel_relation',
    'build_knn_relation',
    'build_radius_relation',
    'build_transposed_kernel_relation',
    'contains_coords',
    'downsample_coords',
    'intersection_coords',
    'inverse_map',
    'kernel_offsets',
    'lookup_coords',
    'sparse_quantize',
    'union_coords',
    'validate_coords',
]
