from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_lattice._native import ext
from mlx_lattice.core import CoordinateMapKey, KernelSpec, SparseTensor
from mlx_lattice.core.types import Triple

__all__ = [
    'conv3d',
    'conv_transpose3d',
    'generative_conv_transpose3d',
    'subm_conv3d',
]


def conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    coordinates: SparseTensor | CoordinateMapKey | mx.array | None = None,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    output_stride = _mul_stride(x.stride, spec.stride)
    if coordinates is None and spec.is_pointwise:
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    if coordinates is not None:
        target_key = _target_key(x, coordinates, output_stride)
        return _relation_conv(
            x,
            _target_weight(weight, spec),
            bias,
            spec,
            map_kind='target',
            output_stride=target_key.stride,
            target_key=target_key,
        )

    return _relation_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='forward',
        output_stride=output_stride,
    )


def subm_conv3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 3,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=1,
        padding=0,
        dilation=dilation,
    )
    _require_odd_kernel(spec.size, 'subm_conv3d')
    if spec.size == (1, 1, 1) and spec.dilation == (1, 1, 1):
        return x.replace(
            feats=_with_bias(_pointwise_features(x, weight), bias)
        )

    return _relation_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='forward',
        output_stride=x.stride,
        reuse_input_coords=True,
    )


def conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> SparseTensor:
    spec = KernelSpec(
        size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return _relation_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='transposed',
        output_stride=_div_stride(x.stride, spec.stride),
    )


def generative_conv_transpose3d(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    kernel_size: int | Sequence[int] = 2,
    stride: int | Sequence[int] = 2,
) -> SparseTensor:
    spec = KernelSpec(size=kernel_size, stride=stride)
    return _relation_conv(
        x,
        weight,
        bias,
        spec,
        map_kind='generative',
        output_stride=_div_stride(x.stride, spec.stride),
    )


# MARK: - execution policy


def _relation_conv(
    x: SparseTensor,
    weight: mx.array,
    bias: mx.array | None,
    spec: KernelSpec,
    *,
    map_kind: str,
    output_stride: Triple,
    reuse_input_coords: bool = False,
    target_key: CoordinateMapKey | None = None,
) -> SparseTensor:
    _validate_feature_dtype(x.feats, weight)
    _validate_metal_coord_dtype(x)
    _validate_weight_for_kernel(x, weight, spec.volume)
    relation = _kernel_relation(x, spec, map_kind, target_key=target_key)
    if relation.n_out_capacity is None or relation.n_kernels is None:
        raise ValueError(
            'kernel relation is missing static shape metadata.'
        )

    feats = ext.sparse_conv_features(
        x.feats,
        weight,
        relation.edges.in_rows,
        relation.edges.out_rows,
        relation.edges.kernel_ids,
        relation.counts,
        relation.row_offsets,
        relation.n_out_capacity,
        relation.n_kernels,
    )
    if reuse_input_coords:
        return x.replace(feats=_with_bias(feats, bias))
    if relation.out_coords is None:
        raise ValueError('kernel relation is missing output coordinates.')
    return SparseTensor(
        relation.out_coords,
        _with_bias(feats, bias),
        stride=output_stride,
        coord_manager=x.coord_manager,
        active_rows=relation.out_count,
    )


def _kernel_relation(
    x: SparseTensor,
    spec: KernelSpec,
    map_kind: str,
    *,
    target_key: CoordinateMapKey | None = None,
):
    if map_kind == 'forward':
        return x.coord_manager.kernel_relation(
            x.coord_key,
            kernel_size=spec.size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
        )
    if map_kind == 'transposed':
        return x.coord_manager.transposed_kernel_relation(
            x.coord_key,
            kernel_size=spec.size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
        )
    if map_kind == 'generative':
        return x.coord_manager.generative_relation(
            x.coord_key,
            kernel_size=spec.size,
            stride=spec.stride,
        )
    if map_kind == 'target':
        if target_key is None:
            raise ValueError('target_key is required for target relations.')
        return x.coord_manager.target_kernel_relation(
            x.coord_key,
            target_key,
            kernel_size=spec.size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
        )
    raise ValueError(
        "map_kind must be 'forward', 'transposed', 'generative', or 'target'."
    )


def _target_key(
    x: SparseTensor,
    coordinates: SparseTensor | CoordinateMapKey | mx.array,
    output_stride: Triple,
) -> CoordinateMapKey:
    if isinstance(coordinates, SparseTensor):
        if coordinates.stride != output_stride:
            raise ValueError(
                'target coordinates must use the convolution output stride.'
            )
        if coordinates.coord_manager is x.coord_manager:
            return coordinates.coord_key
        return x.coord_manager.insert_coords(
            coordinates.coords,
            coordinates.stride,
            coordinates.active_rows,
        )
    if isinstance(coordinates, CoordinateMapKey):
        if not x.coord_manager.owns(coordinates):
            raise ValueError(
                'target coordinate key must belong to x.coord_manager.'
            )
        if coordinates.stride != output_stride:
            raise ValueError(
                'target coordinate key must use the convolution output stride.'
            )
        return coordinates
    return x.coord_manager.insert_coords(coordinates, output_stride)


def _pointwise_features(x: SparseTensor, weight: mx.array) -> mx.array:
    _validate_feature_dtype(x.feats, weight)
    matrix = _pointwise_weight_matrix(x, weight)
    return x.feats @ matrix.T


def _target_weight(weight: mx.array, spec: KernelSpec) -> mx.array:
    if not spec.is_pointwise or weight.ndim != 2:
        return weight
    return mx.expand_dims(weight.T, axis=0)


# MARK: - validation


def _validate_feature_dtype(feats: mx.array, weight: mx.array) -> None:
    if feats.dtype != mx.float32 or weight.dtype != mx.float32:
        raise ValueError('convolution currently supports float32 tensors.')


def _validate_metal_coord_dtype(x: SparseTensor) -> None:
    if mx.default_device() == mx.gpu and x.coords.dtype != mx.int32:
        raise ValueError(
            'Metal sparse convolution requires int32 coordinates.'
        )


def _pointwise_weight_matrix(x: SparseTensor, weight: mx.array) -> mx.array:
    if weight.ndim == 2:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight
    if (
        weight.ndim == 5
        and weight.shape[1] == 1
        and weight.shape[2] == 1
        and weight.shape[3] == 1
    ):
        if weight.shape[4] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[:, 0, 0, 0, :]
    if weight.ndim == 3 and weight.shape[0] == 1:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        return weight[0].T
    raise ValueError(
        'pointwise weight must have shape (C_out, C_in), '
        '(C_out, 1, 1, 1, C_in), or (1, C_in, C_out).'
    )


def _validate_weight_for_kernel(
    x: SparseTensor,
    weight: mx.array,
    kernel_rows: int,
) -> None:
    if weight.ndim == 3:
        if weight.shape[1] != x.channels:
            raise ValueError('weight input channels must match x.channels.')
        if weight.shape[0] != kernel_rows:
            raise ValueError(
                'weight kernel rows must match the convolution kernel.'
            )
        return

    if weight.ndim != 5:
        raise ValueError(
            'mapped convolution weight must have shape (K, C_in, C_out) '
            'or (C_out, Kx, Ky, Kz, C_in).'
        )
    if weight.shape[4] != x.channels:
        raise ValueError('weight input channels must match x.channels.')
    if (
        int(weight.shape[1] * weight.shape[2] * weight.shape[3])
        != kernel_rows
    ):
        raise ValueError(
            'weight kernel rows must match the convolution kernel.'
        )


def _with_bias(feats: mx.array, bias: mx.array | None) -> mx.array:
    if bias is None:
        return feats
    if bias.ndim != 1:
        raise ValueError('bias must have shape (C_out,).')
    if bias.shape[0] != feats.shape[1]:
        raise ValueError('bias channels must match output channels.')
    if bias.dtype != feats.dtype:
        raise ValueError('bias dtype must match output features.')
    return feats + bias


def _require_odd_kernel(values: Triple, op_name: str) -> None:
    if any(value % 2 == 0 for value in values):
        raise ValueError(f'{op_name} requires odd kernel_size values.')


def _mul_stride(lhs: Triple, rhs: Triple) -> Triple:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])


def _div_stride(lhs: Triple, rhs: Triple) -> Triple:
    out = []
    for left, right in zip(lhs, rhs, strict=True):
        if left % right != 0:
            raise ValueError(
                'transpose stride must divide the input tensor stride.'
            )
        out.append(left // right)
    return (out[0], out[1], out[2])
