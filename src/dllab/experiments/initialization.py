"""Zero, large, Xavier, and He initialisation on a tiny MLP.

The estimand is the distribution of hidden activations (and, for zero init,
equality across units) immediately after a forward pass on a fixed Gaussian
input. The DGP is x ~ N(0, I), not a real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dllab._rng import get_rng
from dllab.numpy_net.activations import apply_activation
from dllab.numpy_net.init import gaussian, he_normal, xavier_uniform, zeros


@dataclass(frozen=True)
class InitStudy:
    zero_hidden_std_across_units: float
    zero_units_identical: bool
    large_preact_abs_mean: float
    large_sigmoid_grad_mean: float
    xavier_tanh_hidden_var: float
    he_relu_hidden_var: float


def _hidden_preact(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    bias: NDArray[np.floating],
) -> NDArray[np.floating]:
    return x @ weight + bias


def run_init_study(
    n: int = 64,
    in_features: int = 20,
    hidden: int = 12,
    seed: int | np.random.Generator | None = 2026,
    large_scale: float = 8.0,
) -> InitStudy:
    rng = get_rng(seed)
    x = rng.normal(size=(n, in_features))

    w0, b0 = zeros(in_features, hidden)
    h0 = _hidden_preact(x, w0, b0)
    # With zero weights and zero bias, every unit is the zero function.
    unit_std = float(h0.std(axis=1).mean())
    identical = bool(np.allclose(h0, 0.0))

    w_large, b_large = gaussian(in_features, hidden, scale=large_scale, seed=rng)
    pre_large = _hidden_preact(x, w_large, b_large)
    from dllab.numpy_net.activations import sigmoid_grad

    large_abs = float(np.mean(np.abs(pre_large)))
    large_sgrad = float(np.mean(sigmoid_grad(pre_large)))

    w_x, b_x = xavier_uniform(in_features, hidden, seed=rng)
    h_x = apply_activation("tanh", _hidden_preact(x, w_x, b_x))
    xavier_var = float(np.var(h_x))

    w_h, b_h = he_normal(in_features, hidden, seed=rng)
    h_h = apply_activation("relu", _hidden_preact(x, w_h, b_h))
    he_var = float(np.var(h_h))

    return InitStudy(
        zero_hidden_std_across_units=unit_std,
        zero_units_identical=identical,
        large_preact_abs_mean=large_abs,
        large_sigmoid_grad_mean=large_sgrad,
        xavier_tanh_hidden_var=xavier_var,
        he_relu_hidden_var=he_var,
    )
