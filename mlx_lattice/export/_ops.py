from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, cast

import mlx.core as mx

import mlx_lattice.ops as ops
from mlx_lattice.core import SparseTensor
from mlx_lattice.export.runtime import (
    ExecutionContext,
    GraphValue,
    ParameterBinding,
    iter_value_type_bindings,
    value_type_fields,
)
from mlx_lattice.ir import (
    IRNode,
    IRParameterKind,
    IRValueType,
    op_export_hints,
)
from mlx_lattice.ir.manifest import IRInputRef
from mlx_lattice.nn._export import (
    annotation_is_graph_value,
    annotation_is_value_sequence,
)


class OperationRegistrar(Protocol):
    """Callable registry interface used by operation registration modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Callable: ...

    def binding(self, name: str): ...


def register_operations(lattice_op: OperationRegistrar) -> None:
    """Register semantic aliases and generic public op bindings."""

    register_runtime_ops(lattice_op)
    register_semantic_ops(lattice_op)
    register_public_ops(lattice_op)


def register_runtime_ops(lattice_op: OperationRegistrar) -> None:
    """Register small graph-runtime utility ops."""

    @lattice_op(
        'value.field',
        function=_field_value,
        inputs={'input': 'value'},
        attributes={'field': 'field'},
        output_types={'output': 'any'},
    )
    def _field(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        value = context.input_value(node.inputs['input'])
        field = node.attributes.get('field')
        if not isinstance(field, str):
            raise ValueError(
                f'{node.id}.attributes.field must be a string.'
            )
        return {'output': _field_value(value, field)}


def register_public_ops(lattice_op: OperationRegistrar) -> None:
    """Register generic ``ops.<function>`` bindings from the public ops API."""

    public = public_ops()
    for name, function in public.items():
        signature = inspect.signature(function)
        hints = op_export_hints(function)
        input_args: dict[str, str] = {}
        attr_args: dict[str, str] = {}
        value_attr_args: dict[str, str] = {}
        sequence_inputs: set[str] = set()
        for arg_name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if (
                arg_name in hints.parameters
                or arg_name in hints.optional_parameters
            ):
                continue
            if arg_name in hints.attributes:
                attr_args[arg_name] = arg_name
                continue
            if arg_name in hints.value_attributes:
                value_attr_args[arg_name] = arg_name
                continue
            if parameter.default is inspect.Parameter.empty:
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                    attr_args[arg_name] = arg_name
                elif _is_value_parameter(arg_name, parameter):
                    input_args[arg_name] = arg_name
                    if _is_sequence_parameter(arg_name, parameter):
                        sequence_inputs.add(arg_name)
                else:
                    attr_args[arg_name] = arg_name
            else:
                if _is_value_parameter(arg_name, parameter):
                    value_attr_args[arg_name] = arg_name
                else:
                    attr_args[arg_name] = arg_name

        @lattice_op(
            _op_name(name),
            function=function,
            inputs=input_args,
            attributes=attr_args,
            value_attributes=value_attr_args,
            parameters=_hint_parameters(hints.parameters),
            optional_parameters=_hint_parameters(hints.optional_parameters),
            sequence_inputs=sequence_inputs,
        )
        def _auto(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)


def register_semantic_ops(lattice_op: OperationRegistrar) -> None:
    """Register stable semantic op aliases used by exported artifacts."""

    p = public_ops()

    for op, fn, attrs, value_attrs in (
        (
            'sparse.conv3d',
            p['conv3d'],
            _KERNEL_ATTRS,
            {'coordinates': 'coordinates'},
        ),
        (
            'sparse.subm_conv3d',
            p['subm_conv3d'],
            {'kernel_size': 'kernel_size', 'dilation': 'dilation'},
            {},
        ),
        (
            'sparse.conv_transpose3d',
            p['conv_transpose3d'],
            _KERNEL_ATTRS,
            {},
        ),
        (
            'sparse.generative_conv_transpose3d',
            p['generative_conv_transpose3d'],
            {'kernel_size': 'kernel_size', 'stride': 'stride'},
            {},
        ),
    ):

        @lattice_op(
            op,
            function=fn,
            inputs={'input': 'x'},
            parameters={'weight': 'weight'},
            optional_parameters={'bias': 'bias'},
            attributes=attrs,
            value_attributes=value_attrs,
        )
        def _conv(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    @lattice_op(
        'sparse.add',
        function=p['sparse_add'],
        inputs={'lhs': 'lhs', 'rhs': 'rhs'},
        attributes={'join': 'join'},
    )
    def _sparse_add(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        return passthrough(lattice_op, context, node)

    @lattice_op(
        'feature.linear',
        function=p['linear'],
        inputs={'input': 'x'},
        parameters={'weight': 'weight'},
        optional_parameters={'bias': 'bias'},
    )
    def _linear(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        return passthrough(lattice_op, context, node)

    @lattice_op(
        'feature.quantized_linear',
        function=p['linear'],
        inputs={'input': 'x'},
        parameters={
            'weight': ParameterBinding('weight', 'quantized_weight'),
        },
        optional_parameters={'bias': 'bias'},
    )
    def _quantized_linear(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        return passthrough(lattice_op, context, node)

    for op, fn, attrs, value_attrs in (
        (
            'sparse.quantized_conv3d',
            p['conv3d'],
            _KERNEL_ATTRS,
            {'coordinates': 'coordinates'},
        ),
        (
            'sparse.quantized_subm_conv3d',
            p['subm_conv3d'],
            {'kernel_size': 'kernel_size', 'dilation': 'dilation'},
            {},
        ),
        (
            'sparse.quantized_conv_transpose3d',
            p['conv_transpose3d'],
            _KERNEL_ATTRS,
            {},
        ),
        (
            'sparse.quantized_generative_conv_transpose3d',
            p['generative_conv_transpose3d'],
            {'kernel_size': 'kernel_size', 'stride': 'stride'},
            {},
        ),
    ):

        @lattice_op(
            op,
            function=fn,
            inputs={'input': 'x'},
            parameters={
                'weight': ParameterBinding('weight', 'quantized_weight'),
            },
            optional_parameters={'bias': 'bias'},
            attributes=attrs,
            value_attributes=value_attrs,
        )
        def _quantized_conv(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    for name in (
        'relu',
        'sigmoid',
        'silu',
        'tanh',
        'gelu',
        'leaky_relu',
        'softplus',
        'dropout',
        'batch_norm',
        'layer_norm',
        'rms_norm',
    ):
        signature = inspect.signature(p[name])
        parameter_args = {
            arg: arg
            for arg in ('weight', 'bias', 'mean', 'var')
            if arg in signature.parameters
        }
        attrs = {
            arg: arg
            for arg, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
            and arg not in parameter_args
        }

        @lattice_op(
            f'feature.{name}',
            function=p[name],
            inputs={'input': 'x'},
            optional_parameters=parameter_args,
            attributes=attrs,
        )
        def _feature(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)

    for name in ('sum', 'avg', 'max'):

        @lattice_op(
            f'pool.global_{name}',
            function=p[f'global_{name}_pool'],
            inputs={'input': 'x'},
            handler=_pool_global(lattice_op),
        )
        def _global(
            context: ExecutionContext,
            node: IRNode,
        ) -> dict[str, GraphValue]:
            return passthrough(lattice_op, context, node)


def public_ops() -> dict[str, Callable[..., GraphValue]]:
    """Return public functional ops that can be registered in lattice IR."""

    return {
        name: cast(Callable[..., GraphValue], getattr(ops, name))
        for name in ops.__all__
        if inspect.isfunction(getattr(ops, name))
    }


def field_value_type(value_type: IRValueType, field: str) -> IRValueType:
    """Return the output value type for a supported structural field."""

    fields = value_type_fields(value_type)
    if field not in fields:
        raise ValueError(
            f'field {field!r} is not supported for IR value type '
            f'{value_type!r}.'
        )
    return fields[field]


def passthrough(
    lattice_op: OperationRegistrar,
    context: ExecutionContext,
    node: IRNode,
) -> dict[str, GraphValue]:
    return lattice_op.binding(node.op).run_default(context, node)


def _field_value(value: GraphValue, field: str) -> GraphValue:
    if not _is_allowed_field(value, field):
        raise ValueError(
            f'field {field!r} is not supported for {type(value).__name__}.'
        )
    return cast(GraphValue, getattr(value, field))


def _is_allowed_field(value: GraphValue, field: str) -> bool:
    for binding in iter_value_type_bindings():
        if (
            binding.fields
            and isinstance(value, binding.runtime_type)
            and field in binding.fields
        ):
            return True
    return False


def _pool_global(lattice_op: OperationRegistrar):
    def handler(
        context: ExecutionContext,
        node: IRNode,
    ) -> dict[str, GraphValue]:
        x = context.sparse(_str_ref(node.inputs['input']))
        binding = lattice_op.binding(node.op)
        fn = cast(Callable[[SparseTensor], mx.array], binding.function)
        if x.batch_counts is not None or node.op == 'pool.global_max':
            return {'output': fn(x)}
        if context.batch_size is None:
            raise ValueError(
                f'{node.id} requires batch_counts metadata or a sparse graph '
                'input with known batch_counts.'
            )
        mode = node.op.removeprefix('pool.global_')
        return {
            'output': _global_pool_from_coordinate_batches(
                x,
                context.batch_size,
                mode=cast(Any, mode),
            )
        }

    return handler


def _global_pool_from_coordinate_batches(
    x: SparseTensor,
    batch_size: int,
    *,
    mode: Literal['sum', 'avg'],
) -> mx.array:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    rows = mx.arange(x.capacity, dtype=mx.int32)
    active = (rows < x.active_rows[0]).astype(x.feats.dtype)
    batch_ids = x.coords[:, 0].astype(mx.int32)
    clipped = mx.minimum(mx.maximum(batch_ids, 0), batch_size - 1)
    summed = (
        mx.zeros((batch_size, x.channels), dtype=x.feats.dtype)
        .at[clipped]
        .add(x.feats * active[:, None])
    )
    if mode == 'sum':
        return summed
    counts = (
        mx.zeros((batch_size,), dtype=x.feats.dtype).at[clipped].add(active)
    )
    return summed / mx.maximum(counts, 1)[:, None]


def _str_ref(value: IRInputRef) -> str:
    if not isinstance(value, str):
        raise ValueError('expected a single graph value reference.')
    return value


def _op_name(name: str) -> str:
    return f'ops.{name}'


_KERNEL_ATTRS = {
    'kernel_size': 'kernel_size',
    'stride': 'stride',
    'padding': 'padding',
    'dilation': 'dilation',
}


def _is_sequence_parameter(
    name: str,
    parameter: inspect.Parameter,
) -> bool:
    return annotation_is_value_sequence(parameter.annotation)


def _is_value_parameter(
    name: str,
    parameter: inspect.Parameter,
) -> bool:
    return annotation_is_graph_value(parameter.annotation)


def _hint_parameters(
    hints: Mapping[str, IRParameterKind],
) -> dict[str, ParameterBinding]:
    return {
        name: ParameterBinding(name, _parameter_kind(kind))
        for name, kind in hints.items()
    }


def _parameter_kind(kind: IRParameterKind):
    if kind == 'array_or_quantized_weight':
        return 'array_or_quantized_weight'
    if kind == 'quantized_weight':
        return 'quantized_weight'
    if kind == 'optional_array':
        return 'optional_array'
    return 'array'


_SEMANTIC_ALIAS_FUNCTIONS = {
    'conv3d',
    'subm_conv3d',
    'conv_transpose3d',
    'generative_conv_transpose3d',
    'linear',
    'relu',
    'sigmoid',
    'silu',
    'tanh',
    'gelu',
    'leaky_relu',
    'softplus',
    'dropout',
    'batch_norm',
    'layer_norm',
    'rms_norm',
    'global_sum_pool',
    'global_avg_pool',
    'global_max_pool',
}
