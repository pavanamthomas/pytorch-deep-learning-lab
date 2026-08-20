"""Three-way check: manual reverse mode, finite differences, and autograd.

Used by the flagship note and by ``scripts/run_all.py``. The scalar is
mean squared error of a single affine map composed with tanh — smooth, so
a well-chosen ε can agree with both reverse-mode formulae.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from dllab._rng import get_rng
from dllab.gradcheck.finite_diff import finite_diff_grad, relative_error, sweep_epsilon
from dllab.numpy_net.activations import tanh, tanh_grad
from dllab.numpy_net.losses import mse, mse_grad, naive_softmax, softmax, softmax_cross_entropy


@dataclass(frozen=True)
class ThreeWayCheck:
    loss: float
    rel_fd_vs_analytic: float
    rel_autograd_vs_analytic: float
    rel_fd_vs_autograd: float
    chosen_epsilon: float


def _analytic_affine_tanh_mse_grad(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    pre = x @ weight
    yhat = tanh(pre)
    d_yhat = mse_grad(yhat, y)
    d_pre = d_yhat * tanh_grad(pre)
    d_w = x.T @ d_pre
    return d_w


def run_three_way_check(
    n: int = 6,
    in_features: int = 3,
    out_features: int = 2,
    epsilon: float = 1e-5,
    seed: int | np.random.Generator | None = 2026,
) -> ThreeWayCheck:
    rng = get_rng(seed)
    x = rng.normal(size=(n, in_features))
    y = rng.normal(size=(n, out_features))
    weight = rng.normal(scale=0.4, size=(in_features, out_features))

    def objective(theta: NDArray[np.floating]) -> float:
        w = np.asarray(theta, dtype=float).reshape(in_features, out_features)
        return mse(tanh(x @ w), y)

    analytic = _analytic_affine_tanh_mse_grad(x, weight, y)
    fd = finite_diff_grad(objective, weight, epsilon=epsilon)

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    linear = nn.Linear(in_features, out_features, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight.T)))
    yhat_t = torch.tanh(linear(xt))
    loss_t = torch.mean((yhat_t - yt) ** 2)
    loss_t.backward()
    assert linear.weight.grad is not None
    auto = linear.weight.grad.detach().cpu().numpy().T

    return ThreeWayCheck(
        loss=mse(tanh(x @ weight), y),
        rel_fd_vs_analytic=relative_error(fd, analytic),
        rel_autograd_vs_analytic=relative_error(auto, analytic),
        rel_fd_vs_autograd=relative_error(fd, auto),
        chosen_epsilon=epsilon,
    )


def tanh_mse_epsilon_sweep(
    seed: int | np.random.Generator | None = 2026,
) -> list:
    rng = get_rng(seed)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=(5, 2))
    weight = rng.normal(scale=0.4, size=(3, 2))

    def objective(theta: NDArray[np.floating]) -> float:
        w = np.asarray(theta, dtype=float).reshape(3, 2)
        return mse(tanh(x @ w), y)

    analytic = _analytic_affine_tanh_mse_grad(x, weight, y)
    return sweep_epsilon(objective, analytic, weight)


def ill_scaled_softmax_demo() -> dict[str, float | bool]:
    """Stable vs naive softmax, and CE at huge logits where probabilities saturate."""
    z_ok = np.array([[1.0, 0.0, -0.5]], dtype=float)
    z_bad = np.array([[1000.0, 0.0, -1000.0]], dtype=float)
    labels = np.array([0], dtype=int)
    naive_ok = naive_softmax(z_ok)
    naive_bad = naive_softmax(z_bad)
    stable_bad = softmax(z_bad)
    loss_bad = softmax_cross_entropy(z_bad, labels)
    return {
        "naive_ok_finite": bool(np.all(np.isfinite(naive_ok))),
        "naive_bad_finite": bool(np.all(np.isfinite(naive_bad))),
        "stable_bad_finite": bool(np.all(np.isfinite(stable_bad))),
        "stable_bad_peak": float(stable_bad[0, 0]),
        "ce_at_saturated_logits": loss_bad,
    }
