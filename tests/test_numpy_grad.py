"""Closed-form gradient of a single affine + ReLU + MSE map."""

from __future__ import annotations

import numpy as np

from dllab.numpy_net.layers import Affine
from dllab.numpy_net.activations import relu, relu_grad
from dllab.numpy_net.losses import mse_grad


def _closed_form_affine_relu_mse(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, y: np.ndarray):
    """Written here, not imported from the backward implementation under test."""
    pre = x @ weight + bias
    yhat = np.maximum(pre, 0.0)
    d_yhat = 2.0 * (yhat - y) / float(yhat.size)
    d_pre = d_yhat * (pre > 0.0).astype(float)
    d_weight = x.T @ d_pre
    d_bias = d_pre.sum(axis=0)
    return d_weight, d_bias


def test_affine_relu_mse_matches_closed_form() -> None:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(7, 4))
    y = rng.normal(size=(7, 3))
    # Offset the weights so that pre-activations are unlikely to land on 0.
    weight = rng.normal(size=(4, 3)) + 0.4
    bias = rng.normal(size=(3,))
    dW, db = _closed_form_affine_relu_mse(x, weight, bias, y)

    layer = Affine(weight.copy(), bias.copy())
    pre = layer.forward(x)
    yhat = relu(pre)
    layer.backward(mse_grad(yhat, y) * relu_grad(pre))
    assert layer.grad_weight is not None and layer.grad_bias is not None
    assert np.allclose(layer.grad_weight, dW)
    assert np.allclose(layer.grad_bias, db)
