"""Finite-difference gradient checks against analytic reverse-mode derivatives.

What problem is being solved?
    Independently estimate dL/dθ by probing the scalar L, then compare that
    estimate to the analytic gradient. The same tool shows that the probe
    itself fails when ε is too large (truncation) or too small (cancellation).

What assumptions are required?
    L is a differentiable function of a real array, at least twice, in a
    neighbourhood of the point being checked. Central differences use
    L(θ+ε e_i) and L(θ-ε e_i). ReLU at 0 is not differentiable.

Why was this method chosen?
    Central differences have truncation error O(ε²) from the Taylor remainder
    and cancellation error O(ε_mach / ε) from subtracting nearly equal
    floats. A sweep in ε makes that trade-off visible.

What alternative method could have been used?
    Forward differences (O(ε) truncation); complex-step derivatives
    (excellent for holomorphic real-on-real functions); autograd as the
    only check.

What can go wrong?
    Too-large ε: the O(ε²) remainder dominates. Too-small ε: θ+ε and θ
    are not distinct in floating point, or L(θ+ε)-L(θ-ε) loses all digits.
    At a kink, the finite-difference quotient converges to a number that
    need not equal the autograd convention.

How is correctness independently checked?
    On a smooth affine+tanh+MSE map, relative error is small at ε ≈ 1e-5
    or 1e-6 in float64, and larger at both ends of the sweep. ReLU at 0
    is a designed disagreement.

What can legitimately be concluded?
    Agreement at a good ε is evidence that the analytic gradient matches
    the scalar actually computed. Disagreement at extreme ε is a property
    of floating-point calculus, not necessarily a bug in the backward pass.

What cannot be concluded?
    That a network is correctly implemented for every input, or that
    autograd is exact in float32 on an ill-scaled softmax.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def relative_error(a: NDArray[np.floating], b: NDArray[np.floating], eps: float = 1e-12) -> float:
    """||a-b||_2 / max(||a||_2 + ||b||_2, eps)."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("arrays must have the same shape")
    num = float(np.linalg.norm(aa - bb))
    den = float(np.linalg.norm(aa) + np.linalg.norm(bb))
    return num / max(den, eps)


def finite_diff_grad(
    f: Callable[[NDArray[np.floating]], float],
    x: NDArray[np.floating],
    epsilon: float = 1e-5,
) -> NDArray[np.floating]:
    """Central-difference gradient of a scalar map f: R^{shape} -> R."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    theta = np.asarray(x, dtype=float).copy()
    grad = np.zeros_like(theta, dtype=float)
    for idx in np.ndindex(theta.shape):
        original = theta[idx]
        theta[idx] = original + epsilon
        plus = float(f(theta))
        theta[idx] = original - epsilon
        minus = float(f(theta))
        theta[idx] = original
        grad[idx] = (plus - minus) / (2.0 * epsilon)
    return grad


@dataclass(frozen=True)
class EpsilonSweepPoint:
    epsilon: float
    relative_error: float
    max_abs_error: float


def sweep_epsilon(
    f: Callable[[NDArray[np.floating]], float],
    analytic_grad: NDArray[np.floating],
    x: NDArray[np.floating],
    epsilons: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
) -> list[EpsilonSweepPoint]:
    """Relative error of the central-difference gradient versus analytic, vs ε."""
    g_ref = np.asarray(analytic_grad, dtype=float)
    points: list[EpsilonSweepPoint] = []
    for eps in epsilons:
        g_fd = finite_diff_grad(f, x, epsilon=eps)
        points.append(
            EpsilonSweepPoint(
                epsilon=float(eps),
                relative_error=relative_error(g_fd, g_ref),
                max_abs_error=float(np.max(np.abs(g_fd - g_ref))),
            )
        )
    return points


def relu_kink_central_difference(epsilon: float = 1e-5) -> dict[str, float]:
    """Central difference of relu at 0 is 1/2 for any ε > 0 that does not overflow.

    relu(ε)=ε, relu(-ε)=0, so (ε - 0)/(2ε) = 1/2.
    Autograd and this laboratory use relu'(0) = 0.
    The disagreement does not vanish as ε -> 0. That is cancellation vs
    truncation? Neither: it is a non-differentiable point.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def f(t: NDArray[np.floating]) -> float:
        v = float(np.asarray(t, dtype=float).reshape(-1)[0])
        return float(np.maximum(v, 0.0))

    x = np.array([0.0], dtype=float)
    g_fd = finite_diff_grad(f, x, epsilon=epsilon)
    return {
        "epsilon": float(epsilon),
        "central_difference": float(g_fd[0]),
        "autograd_convention": 0.0,
        "forward_difference": 1.0,
        "backward_difference": 0.0,
    }
