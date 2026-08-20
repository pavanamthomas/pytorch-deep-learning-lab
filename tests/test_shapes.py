"""Shape contracts for affine, MLP, conv1d, and the PyTorch mirrors."""

from __future__ import annotations

import numpy as np
import torch

from dllab.numpy_net.layers import Affine, conv1d_backward, conv1d_forward, conv1d_toeplitz
from dllab.numpy_net.mlp import NumpyMLP
from dllab.torch_net.cnn import MotifCNN
from dllab.torch_net.mlp import TorchMLP


def test_affine_forward_backward_shapes() -> None:
    x = np.zeros((4, 3))
    layer = Affine(np.zeros((3, 5)), np.zeros(5))
    y = layer.forward(x)
    assert y.shape == (4, 5)
    dx = layer.backward(np.ones_like(y))
    assert dx.shape == (4, 3)
    assert layer.grad_weight is not None and layer.grad_weight.shape == (3, 5)
    assert layer.grad_bias is not None and layer.grad_bias.shape == (5,)


def test_mlp_logit_shape() -> None:
    net = NumpyMLP.from_sizes([2, 6, 3], activation="relu", seed=2026)
    logits = net.forward(np.zeros((5, 2)))
    assert logits.shape == (5, 3)


def test_torch_mlp_logit_shape() -> None:
    model = TorchMLP([2, 6, 3], activation="relu")
    out = model(torch.zeros(5, 2))
    assert tuple(out.shape) == (5, 3)


def test_conv1d_shapes_and_toeplitz() -> None:
    x = np.random.default_rng(2026).normal(size=(2, 1, 10))
    w = np.random.default_rng(1).normal(size=(1, 1, 3))
    b = np.zeros(1)
    y = conv1d_forward(x, w, b)
    assert y.shape == (2, 1, 8)
    dx, dw, db = conv1d_backward(x, w, np.ones_like(y))
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == (1,)
    toe = conv1d_toeplitz(w, length=10)
    assert toe.shape == (8, 10)
    # Tied diagonals: every window uses the same three coefficients.
    assert np.allclose(toe[0, 0:3], w[0, 0])
    assert np.allclose(toe[1, 1:4], w[0, 0])
    single = x[:1]
    y_single = conv1d_forward(single, w).reshape(-1)
    assert np.allclose(y_single, toe @ single.reshape(-1))


def test_motif_cnn_shape() -> None:
    model = MotifCNN(n_filters=4, kernel_size=3)
    x = torch.zeros(3, 1, 16)
    logits = model(x)
    assert tuple(logits.shape) == (3, 2)
