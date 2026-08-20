"""Saturation of sigmoid/tanh and a dead ReLU unit.

The estimand is the mean local derivative on a Gaussian pre-activation
(saturation) and the incoming-weight gradient of a ReLU unit whose bias
is so negative that every example in the batch is clipped (dead ReLU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dllab._rng import get_rng
from dllab.numpy_net.activations import relu, relu_grad, sigmoid_grad, tanh_grad
from dllab.numpy_net.layers import Affine
from dllab.numpy_net.losses import mse_grad


@dataclass(frozen=True)
class ActivationStudy:
    sigmoid_grad_at_scale_1: float
    sigmoid_grad_at_scale_8: float
    tanh_grad_at_scale_1: float
    tanh_grad_at_scale_8: float
    dead_relu_weight_grad_abs_max: float
    dead_relu_fraction_zero: float


def run_activation_study(n: int = 64, dim: int = 8, seed: int | np.random.Generator | None = 2026) -> ActivationStudy:
    rng = get_rng(seed)
    z1 = rng.normal(scale=1.0, size=(n, dim))
    z8 = rng.normal(scale=8.0, size=(n, dim))

    # Dead ReLU: large negative bias, modest weights, so pre-activations are < 0.
    w = rng.normal(scale=0.05, size=(dim, 1))
    b = np.array([-3.0])
    layer = Affine(w, b)
    x = rng.normal(size=(n, dim))
    pre = layer.forward(x)
    hidden = relu(pre)
    # MSE to a nonzero target so the upstream gradient is not itself zero.
    y = np.ones_like(hidden)
    loss_grad = mse_grad(hidden, y)
    dpre = loss_grad * relu_grad(pre)
    _ = layer.backward(dpre)
    assert layer.grad_weight is not None

    return ActivationStudy(
        sigmoid_grad_at_scale_1=float(np.mean(sigmoid_grad(z1))),
        sigmoid_grad_at_scale_8=float(np.mean(sigmoid_grad(z8))),
        tanh_grad_at_scale_1=float(np.mean(tanh_grad(z1))),
        tanh_grad_at_scale_8=float(np.mean(tanh_grad(z8))),
        dead_relu_weight_grad_abs_max=float(np.max(np.abs(layer.grad_weight))),
        dead_relu_fraction_zero=float(np.mean(hidden == 0.0)),
    )
