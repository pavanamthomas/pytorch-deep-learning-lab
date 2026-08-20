"""Multilayer perceptron with manual reverse-mode backpropagation.

What problem is being solved?
    Compose affine maps and elementwise activations into a function
    R^{N x D} -> R^{N x C} whose derivatives with respect to every weight
    can be computed by the chain rule, with no autograd tape.

What assumptions are required?
    Hidden layers share one activation. The last layer is linear (logits
    or a regression head). Mini-batches are iid draws from whatever DGP
    the caller supplies. Parameters use the NumPy layout (in, out).

Why was this method chosen?
    A fully connected stack is the smallest architecture in which vanishing
    gradients, dead ReLU, and initialisation symmetry are already visible.

What alternative method could have been used?
    A single affine model; a convolution; a PyTorch ``nn.Module`` from the
    start.

What can go wrong?
    Zero init; saturation; exploding products of Jacobians; label leakage
    if the same rows are used for selection and evaluation (that is a data
    issue, not an MLP issue).

How is correctness independently checked?
    Closed-form affine+ReLU+MSE gradients; finite differences on a smooth
    network; agreement with a PyTorch MLP given identical weights and batch.

What can legitimately be concluded?
    Forward values and reverse-mode gradients of this architecture match
    the algebra in ``docs/backpropagation.md`` on the tests in ``tests/``.

What cannot be concluded?
    That the MLP is a consistent estimator of a population risk, or that
    a decision boundary on two-moons transfers to any other DGP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dllab._rng import get_rng
from dllab.numpy_net.init import init_affine
from dllab.numpy_net.layers import Activation, Affine
from dllab.numpy_net.losses import (
    mse,
    mse_grad,
    softmax_cross_entropy,
    softmax_cross_entropy_grad,
)
from dllab.numpy_net.sgd import sgd_step


@dataclass
class NumpyMLP:
    """Fully connected stack. Hidden activations; linear last layer."""

    layers: list[Affine]
    activations: list[Activation]
    activation_name: str

    @classmethod
    def from_sizes(
        cls,
        sizes: list[int],
        activation: str = "relu",
        init: str = "he",
        seed: int | np.random.Generator | None = 2026,
        scale: float = 1.0,
    ) -> NumpyMLP:
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least in_features and out_features")
        if any(s < 1 for s in sizes):
            raise ValueError("all sizes must be positive")
        rng = get_rng(seed)
        layers: list[Affine] = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:], strict=True):
            w, b = init_affine(in_f, out_f, scheme=init, seed=rng, scale=scale)
            layers.append(Affine(w, b))
        n_hidden = len(layers) - 1
        acts = [Activation(activation) for _ in range(n_hidden)]
        return cls(layers=layers, activations=acts, activation_name=activation)

    @property
    def sizes(self) -> list[int]:
        out = [self.layers[0].in_features]
        out.extend(layer.out_features for layer in self.layers)
        return out

    def parameters(self) -> list[NDArray[np.floating]]:
        params: list[NDArray[np.floating]] = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)
        return params

    def grads(self) -> list[NDArray[np.floating]]:
        out: list[NDArray[np.floating]] = []
        for layer in self.layers:
            if layer.grad_weight is None or layer.grad_bias is None:
                raise ValueError("backward has not populated gradients")
            out.append(layer.grad_weight)
            out.append(layer.grad_bias)
        return out

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        h = np.asarray(x, dtype=float)
        if h.ndim != 2:
            raise ValueError("x must have shape (batch, in_features)")
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = self.activations[i].forward(h)
        return h

    def backward(self, grad_output: NDArray[np.floating]) -> NDArray[np.floating]:
        dy = np.asarray(grad_output, dtype=float)
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                dy = self.activations[i].backward(dy)
            dy = self.layers[i].backward(dy)
        return dy

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        sgd_step(self.parameters(), self.grads(), lr=lr, weight_decay=weight_decay)


def mse_backward_pass(net: NumpyMLP, x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Forward, MSE, backward. Returns the scalar loss."""
    yhat = net.forward(x)
    loss = mse(yhat, y)
    net.backward(mse_grad(yhat, y))
    return loss


def softmax_ce_backward_pass(
    net: NumpyMLP,
    x: NDArray[np.floating],
    labels: NDArray[np.integer],
    class_weight: NDArray[np.floating] | None = None,
) -> float:
    logits = net.forward(x)
    loss = softmax_cross_entropy(logits, labels, class_weight=class_weight)
    net.backward(softmax_cross_entropy_grad(logits, labels, class_weight=class_weight))
    return loss


def train_softmax_ce(
    net: NumpyMLP,
    x: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    lr: float,
    epochs: int,
    class_weight: NDArray[np.floating] | None = None,
    weight_decay: float = 0.0,
) -> list[float]:
    """Full-batch CE training. Tiny experiments only."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    history: list[float] = []
    for _ in range(epochs):
        loss = softmax_ce_backward_pass(net, x, labels, class_weight=class_weight)
        net.step(lr=lr, weight_decay=weight_decay)
        history.append(loss)
    return history
