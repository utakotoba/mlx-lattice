from __future__ import annotations

from typing import Any, cast

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    conv3d,
    conv_transpose3d,
    generative_conv_transpose3d,
    subm_conv3d,
)
from tests.cases.types import Tolerance, ValueCase
from tests.support import active_coords, active_feats, mx


def cases() -> list[ValueCase]:
    relaxed = Tolerance(abs=2e-2, rel=2e-2)
    return [
        ValueCase('conv3d_modes', _conv3d_modes),
        ValueCase('conv3d_gradients', _conv3d_gradients),
        ValueCase('conv3d_mode_gradients', _conv3d_mode_gradients),
        ValueCase('conv3d_jvp', _conv3d_jvp),
        ValueCase('conv3d_active_rows', _conv3d_active_rows),
        ValueCase('conv3d_float16', _conv3d_float16, tolerance=relaxed),
        ValueCase(
            'conv3d_dense_c16_forward',
            _dense_forward(16),
            tolerance=relaxed,
        ),
        ValueCase(
            'conv3d_dense_c32_weight_grad',
            _dense_weight_grad(32),
            tolerance=relaxed,
        ),
        ValueCase(
            'conv3d_dense_c64_weight_grad',
            _dense_weight_grad(64),
            tolerance=relaxed,
        ),
        ValueCase(
            'conv3d_asymmetric_backward_16_64',
            _asymmetric_backward(16, 64),
            tolerance=relaxed,
        ),
        ValueCase(
            'conv3d_asymmetric_backward_64_16',
            _asymmetric_backward(64, 16),
            tolerance=relaxed,
        ),
    ]


def _line_inputs(
    dtype: Any = mx.float32,
) -> tuple[mx.array, mx.array, mx.array]:
    coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [3.0]], dtype=dtype)
    weight = mx.array([1.0, 2.0, 3.0], dtype=dtype).reshape(1, 3, 1, 1, 1)
    return coords, feats, weight


def _conv3d_modes() -> object:
    coords, feats, weight = _line_inputs()
    x = SparseTensor(coords, feats)
    generic = conv3d(x, weight, kernel_size=(3, 1, 1))
    subm = subm_conv3d(x, mx.ones_like(weight), kernel_size=(3, 1, 1))

    transposed_input = SparseTensor(
        mx.array([[0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[4.0]], dtype=mx.float32),
        stride=(2, 1, 1),
    )
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1, 2, 1, 1, 1
    )
    transposed = conv_transpose3d(
        transposed_input,
        transpose_weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    generated = generative_conv_transpose3d(
        transposed_input,
        transpose_weight,
        kernel_size=(2, 1, 1),
        stride=(2, 1, 1),
    )
    mx.eval(
        generic.coords,
        generic.feats,
        generic.active_rows,
        subm.feats,
        transposed.coords,
        transposed.feats,
        generated.coords,
        generated.feats,
    )
    return (
        cast('list[list[float]]', active_feats(generic).tolist()),
        cast('list[list[float]]', subm.feats.tolist()),
        active_coords(transposed),
        cast('list[list[float]]', active_feats(transposed).tolist()),
        active_coords(generated),
        cast('list[list[float]]', active_feats(generated).tolist()),
    )


def _conv3d_gradients() -> object:
    coords, feats, weight = _line_inputs()

    def loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats)

    grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(feats, weight)
    mx.eval(grad_feats, grad_weight)
    return grad_feats.tolist(), grad_weight.tolist()


def _conv3d_mode_gradients() -> object:
    coords, feats, weight = _line_inputs()

    def subm_loss(feats_arg: mx.array, weight_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return mx.sum(
            subm_conv3d(x, weight_arg, kernel_size=(3, 1, 1)).feats
        )

    subm_grad_feats, subm_grad_weight = mx.grad(
        subm_loss,
        argnums=(0, 1),
    )(feats, weight)

    transpose_coords = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    transpose_feats = mx.array([[4.0]], dtype=mx.float32)
    transpose_weight = mx.array([2.0, 3.0], dtype=mx.float32).reshape(
        1, 2, 1, 1, 1
    )

    def transpose_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    def generative_loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(transpose_coords, feats_arg, stride=(2, 1, 1))
        return mx.sum(
            generative_conv_transpose3d(
                x,
                weight_arg,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
            ).feats
        )

    transpose_grad_feats, transpose_grad_weight = mx.grad(
        transpose_loss,
        argnums=(0, 1),
    )(transpose_feats, transpose_weight)
    generative_grad_feats, generative_grad_weight = mx.grad(
        generative_loss,
        argnums=(0, 1),
    )(transpose_feats, transpose_weight)
    mx.eval(
        subm_grad_feats,
        subm_grad_weight,
        transpose_grad_feats,
        transpose_grad_weight,
        generative_grad_feats,
        generative_grad_weight,
    )
    return (
        subm_grad_feats.tolist(),
        subm_grad_weight.tolist(),
        transpose_grad_feats.tolist(),
        transpose_grad_weight.tolist(),
        generative_grad_feats.tolist(),
        generative_grad_weight.tolist(),
    )


def _conv3d_jvp() -> object:
    coords, feats, weight = _line_inputs()
    tangent = mx.ones_like(feats)

    def features(feats_arg: mx.array) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return conv3d(x, weight, kernel_size=(3, 1, 1)).feats

    _, jvps = mx.jvp(features, [feats], [tangent])
    mx.eval(jvps[0])
    return jvps[0].tolist()


def _conv3d_active_rows() -> object:
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 99, 0, 0],
            [0, 100, 0, 0],
        ],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0], [2.0], [100.0], [200.0]], dtype=mx.float32)
    x = SparseTensor(
        coords,
        feats,
        active_rows=mx.array([2], dtype=mx.int32),
    )
    weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)
    out = conv3d(x, weight, kernel_size=(3, 1, 1))
    mx.eval(out.coords, out.feats, out.active_rows)
    return (
        active_coords(out),
        cast('list[list[float]]', active_feats(out).tolist()),
        cast('list[int]', out.active_rows.tolist()),
    )


def _conv3d_float16() -> object:
    coords, feats, weight = _line_inputs(mx.float16)
    x = SparseTensor(coords, feats)
    out = conv3d(x, weight, kernel_size=(3, 1, 1))
    mx.eval(out.feats)
    assert out.feats.dtype == mx.float16
    return active_feats(out).astype(mx.float32).tolist()


def _dense_forward(channels: int):
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [
            ((row + 1) * (channel + 3) % 37) / 37.0
            for channel in range(channels)
        ]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0
        for index in range(channels * 3 * 3 * 3 * channels)
    ]

    def run() -> object:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            channels, 3, 3, 3, channels
        )
        x = SparseTensor(coords, feats)
        out = conv3d(x, weight, kernel_size=3).feats
        mx.eval(out)
        return cast('list[Any]', out.tolist())

    return run


def _dense_weight_grad(channels: int):
    rows = 32768
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [
            ((row + 1) * (channel + 3) % 37) / 37.0
            for channel in range(channels)
        ]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0
        for index in range(channels * 3 * 3 * 3 * channels)
    ]

    def run() -> object:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            channels, 3, 3, 3, channels
        )

        def loss(weight_arg: mx.array) -> mx.array:
            x = SparseTensor(coords, feats)
            return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

        grad = mx.grad(loss)(weight)
        mx.eval(grad)
        return grad.tolist()

    return run


def _asymmetric_backward(channels_in: int, channels_out: int):
    rows = 4096
    coords_values = [
        [0, row % 97, (row // 97) % 97, row // (97 * 97)]
        for row in range(rows)
    ]
    feats_values = [
        [
            ((row + 1) * (channel + 3) % 37) / 37.0
            for channel in range(channels_in)
        ]
        for row in range(rows)
    ]
    weight_values = [
        ((index % 23) - 11) / 23.0
        for index in range(channels_out * 3 * 3 * 3 * channels_in)
    ]

    def run() -> object:
        coords = mx.array(coords_values, dtype=mx.int32)
        feats = mx.array(feats_values, dtype=mx.float32)
        weight = mx.array(weight_values, dtype=mx.float32).reshape(
            channels_out, 3, 3, 3, channels_in
        )

        def loss(
            feats_arg: mx.array,
            weight_arg: mx.array,
        ) -> mx.array:
            x = SparseTensor(coords, feats_arg)
            return mx.sum(conv3d(x, weight_arg, kernel_size=3).feats)

        grad_feats, grad_weight = mx.grad(loss, argnums=(0, 1))(
            feats,
            weight,
        )
        mx.eval(grad_feats, grad_weight)
        return grad_feats.tolist(), grad_weight.tolist()

    return run
