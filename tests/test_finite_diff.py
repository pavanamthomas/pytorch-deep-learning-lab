"""Finite-difference agreement at a good epsilon, and designed failures at the ends."""

from __future__ import annotations

import numpy as np

from dllab.gradcheck.finite_diff import finite_diff_grad, relative_error, relu_kink_central_difference, sweep_epsilon
from dllab.gradcheck.three_way import run_three_way_check, tanh_mse_epsilon_sweep
from dllab.numpy_net.activations import tanh, tanh_grad
from dllab.numpy_net.losses import mse, mse_grad


def test_finite_diff_agrees_at_good_epsilon() -> None:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=(5, 2))
    weight = rng.normal(scale=0.5, size=(3, 2))

    def objective(theta: np.ndarray) -> float:
        w = theta.reshape(3, 2)
        return mse(tanh(x @ w), y)

    pre = x @ weight
    analytic = x.T @ (mse_grad(tanh(pre), y) * tanh_grad(pre))
    fd = finite_diff_grad(objective, weight, epsilon=1e-5)
    assert relative_error(fd, analytic) < 1e-6


def test_three_way_check_agrees() -> None:
    result = run_three_way_check(seed=2026, epsilon=1e-5)
    assert result.rel_fd_vs_analytic < 1e-6
    assert result.rel_autograd_vs_analytic < 1e-10
    assert result.rel_fd_vs_autograd < 1e-6


def test_epsilon_sweep_u_shape() -> None:
    points = tanh_mse_epsilon_sweep(seed=2026)
    by_eps = {p.epsilon: p.relative_error for p in points}
    good = by_eps[1e-5]
    assert good < by_eps[1e-1]
    assert good < by_eps[1e-10]


def test_relu_kink_central_difference_is_half() -> None:
    result = relu_kink_central_difference(epsilon=1e-6)
    assert abs(result["central_difference"] - 0.5) < 1e-12
    assert result["autograd_convention"] == 0.0
