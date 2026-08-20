"""Elementwise activations and their local derivatives.

What problem is being solved?
    Map a pre-activation to a hidden representation, and return the Jacobian
    diagonal needed for reverse-mode backpropagation.

What assumptions are required?
    Inputs are real arrays. ReLU is treated as having derivative 0 at 0,
    matching the conventional autograd choice, not the full subdifferential.

Why was this method chosen?
    Closed-form local derivatives make the chain rule visible. GELU uses the
    erf definition so that a NumPy forward pass can match ``torch.nn.GELU``.

What alternative method could have been used?
    Automatic differentiation; a piecewise-linear approximation to GELU;
    leaving the subgradient of ReLU at 0 unspecified.

What can go wrong?
    Sigmoid and tanh saturate: local derivatives vanish. ReLU can die.
    Finite differences at the ReLU kink do not recover the autograd value.

How is correctness independently checked?
    Tests compare analytic derivatives to central differences away from kinks,
    and compare affine+ReLU+MSE gradients to a closed form written in the test.

What can legitimately be concluded?
    The local derivative formulae used in this laboratory are the ones
    implemented here, at points of differentiability.

What cannot be concluded?
    That a particular activation is appropriate for an empirical task.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_SQRT_2 = float(np.sqrt(2.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logistic function, evaluated in a numerically stable way.

    For x >= 0, 1/(1+e^{-x}). For x < 0, e^{x}/(1+e^{x}), which avoids
    overflow of exp(-x) when x is a large negative number.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def sigmoid_grad(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """d/dx sigmoid(x) = s(1-s). Saturates to 0 for |x| large."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.tanh(np.asarray(x, dtype=float))


def tanh_grad(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """d/dx tanh(x) = 1 - tanh(x)^2."""
    t = tanh(x)
    return 1.0 - t * t


def relu(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def relu_grad(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Indicator (x > 0). At 0 the conventional autograd value is 0."""
    return (np.asarray(x, dtype=float) > 0.0).astype(float)


def gelu(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exact GELU: x * Phi(x) = 0.5 x (1 + erf(x / sqrt(2)))."""
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.erf(x / _SQRT_2))


def gelu_grad(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """d/dx GELU(x) = Phi(x) + x phi(x)."""
    x = np.asarray(x, dtype=float)
    cdf = 0.5 * (1.0 + np.erf(x / _SQRT_2))
    pdf = np.exp(-0.5 * x * x) / _SQRT_2PI
    return cdf + x * pdf


_TABLE: dict[str, tuple[object, object]] = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "gelu": (gelu, gelu_grad),
}


def get_activation(name: str) -> tuple[object, object]:
    key = name.lower()
    if key not in _TABLE:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(_TABLE)}")
    return _TABLE[key]


def apply_activation(name: str, x: NDArray[np.floating]) -> NDArray[np.floating]:
    fn, _ = get_activation(name)
    return fn(x)  # type: ignore[operator]


def apply_activation_grad(name: str, x: NDArray[np.floating]) -> NDArray[np.floating]:
    _, dfn = get_activation(name)
    return dfn(x)  # type: ignore[operator]
