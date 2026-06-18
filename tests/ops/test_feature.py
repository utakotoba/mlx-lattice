from __future__ import annotations

from typing import Any, cast

import pytest

from mlx_lattice import SparseTensor
from mlx_lattice.ops import (
    batch_norm,
    dropout,
    gelu,
    layer_norm,
    leaky_relu,
    linear,
    relu,
    rms_norm,
    sigmoid,
    silu,
    softplus,
    tanh,
)
from tests.support import (
    assert_nested_close,
    assert_same_sparse_identity,
    mx,
)

pytestmark = [
    pytest.mark.ops,
    pytest.mark.feature,
    pytest.mark.usefixtures('selected_backend'),
]


def _tensor() -> SparseTensor:
    return SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[-1.0, 2.0], [3.0, -4.0]], dtype=mx.float32),
    )


def test_linear_matches_dense_reference_and_preserves_sparse_identity() -> (
    None
):
    x = _tensor()
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    out = linear(x, weight, bias)

    assert out.feats.tolist() == [[5.0, 8.0], [-5.0, -14.0]]
    assert_same_sparse_identity(out, x)


def test_activation_feature_ops_keep_coordinate_contract() -> None:
    x = _tensor()

    assert relu(x).feats.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    leak = cast(
        'list[list[float]]',
        leaky_relu(x, negative_slope=0.1).feats.tolist(),
    )
    assert_nested_close(leak, [[-0.1, 2.0], [3.0, -0.4]])

    for out in [sigmoid(x), silu(x), softplus(x), gelu(x), tanh(x)]:
        assert out.feats.shape == x.feats.shape
        assert_same_sparse_identity(out, x)


def test_dropout_eval_is_identity_and_training_masks_only_features() -> (
    None
):
    x = _tensor()

    eval_out = dropout(x, training=False)
    train_out = dropout(x, p=0.5)

    assert eval_out.feats.tolist() == x.feats.tolist()
    assert_same_sparse_identity(eval_out, x)
    assert train_out.feats.shape == x.feats.shape
    assert_same_sparse_identity(train_out, x)


def test_normalization_feature_ops_apply_affine_parameters() -> None:
    x = SparseTensor(
        mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32),
        mx.array([[1.0, 3.0], [3.0, 7.0]], dtype=mx.float32),
    )
    weight = mx.array([2.0, 3.0], dtype=mx.float32)
    bias = mx.array([0.5, -0.5], dtype=mx.float32)

    bn = batch_norm(
        x,
        weight=weight,
        bias=bias,
        mean=mx.array([2.0, 5.0], dtype=mx.float32),
        var=mx.array([1.0, 4.0], dtype=mx.float32),
        eps=1e-12,
    )
    ln = layer_norm(x, weight=weight, bias=bias, eps=1e-12)
    rms = rms_norm(x, weight=weight, eps=1e-12)

    assert bn.feats.tolist() == [[-1.5, -3.5], [2.5, 2.5]]
    assert ln.feats.shape == x.feats.shape
    assert rms.feats.shape == x.feats.shape
    assert_same_sparse_identity(bn, x)
    assert_same_sparse_identity(ln, x)
    assert_same_sparse_identity(rms, x)


def _feature_transform_case():
    coords = mx.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.int32)
    feats = mx.array([[-1.0, 2.0], [3.0, -4.0]], dtype=mx.float32)
    weight = mx.array([[2.0, 3.0], [5.0, 7.0]], dtype=mx.float32)
    bias = mx.array([1.0, -1.0], dtype=mx.float32)

    def features(
        feats_arg: mx.array,
        weight_arg: mx.array,
        bias_arg: mx.array,
    ) -> mx.array:
        x = SparseTensor(coords, feats_arg)
        return relu(linear(x, weight_arg, bias_arg)).feats

    def loss(
        feats_arg: mx.array,
        weight_arg: mx.array,
        bias_arg: mx.array,
    ) -> mx.array:
        return mx.sum(features(feats_arg, weight_arg, bias_arg))

    return feats, weight, bias, features, loss


def test_feature_ops_support_grad_vjp_and_jvp() -> None:
    feats, weight, bias, features, loss = _feature_transform_case()
    grad_feats, grad_weight, grad_bias = mx.grad(
        loss,
        argnums=(0, 1, 2),
    )(feats, weight, bias)
    outputs, vjps = mx.vjp(
        features,
        [feats, weight, bias],
        [mx.ones((2, 2), dtype=mx.float32)],
    )
    _, jvps = mx.jvp(
        features,
        [feats, weight, bias],
        [mx.ones_like(feats), mx.ones_like(weight), mx.ones_like(bias)],
    )
    assert outputs[0].tolist() == [[5.0, 8.0], [0.0, 0.0]]
    assert grad_feats.tolist() == [[7.0, 10.0], [0.0, 0.0]]
    assert grad_weight.tolist() == [[-1.0, 2.0], [-1.0, 2.0]]
    assert grad_bias.tolist() == [1.0, 1.0]
    assert [value.tolist() for value in vjps] == [
        [[7.0, 10.0], [0.0, 0.0]],
        [[-1.0, 2.0], [-1.0, 2.0]],
        [1.0, 1.0],
    ]
    assert jvps[0].tolist() == [[7.0, 14.0], [0.0, 0.0]]


def test_feature_ops_support_mx_compile(compile_backend) -> None:
    feats, weight, bias, features, _ = _feature_transform_case()

    assert mx.compile(features)(feats, weight, bias).tolist() == [
        [5.0, 8.0],
        [0.0, 0.0],
    ]


def test_feature_ops_reject_invalid_contracts() -> None:
    x = _tensor()

    with pytest.raises(ValueError, match='weight'):
        linear(x, mx.ones((2, 3), dtype=mx.float32))
    with pytest.raises(ValueError, match='bias'):
        linear(x, mx.ones((1, 2), dtype=mx.float32), mx.ones((2, 1)))
    with pytest.raises(ValueError, match='approximate'):
        gelu(x, approximate=cast('Any', 'unsupported'))
    with pytest.raises(ValueError, match='p must'):
        dropout(x, p=1.0)
    with pytest.raises(ValueError, match='eps'):
        layer_norm(x, eps=0)
