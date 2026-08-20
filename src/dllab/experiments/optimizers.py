"""SGD, momentum, Adam, and AdamW on a 2-D quadratic, plus coupled vs decoupled decay.

What problem is being solved?
    Compare first-order methods on a known convex quadratic, then show that
    Adam with L2 in the gradient is not the same update as AdamW.

What assumptions are required?
    The quadratic is f(w) = 0.5 (w1^2 / a + a w2^2) with a > 1 (ill-conditioned).
    Weight-decay comparisons use the same numerical coefficient λ on that
    quadratic plus an explicit L2 term or a decoupled decay.

Why was this method chosen?
    A two-parameter objective can be plotted. The curvature mismatch is
    large enough that momentum and Adam leave a visible trace relative to
    SGD, without requiring a deep net.

What alternative method could have been used?
    A Rosenbrock banana; a tiny MLP; a closed-form Newton step as the
    comparator.

What can go wrong?
    Reading a faster drop in f as a general ranking of optimisers.
    Equating Adam's ``weight_decay`` argument in older APIs with AdamW.

How is correctness independently checked?
    The gradient of the quadratic is written in closed form. After one
    SGD step, the parameter matches w - lr * grad. Adam vs AdamW parameter
    trajectories differ once λ > 0 and the two coordinates have unequal
    gradient scales.

What can legitimately be concluded?
    On this quadratic, the four methods produce the recorded paths.
    Coupled L2 inside Adam scales the decay by the adaptive denominator;
    AdamW does not.

What cannot be concluded?
    That AdamW is uniformly preferable, or that these paths describe
    ImageNet training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class QuadraticPaths:
    sgd: np.ndarray
    momentum: np.ndarray
    adam: np.ndarray
    adamw: np.ndarray
    values_sgd: np.ndarray
    values_adam: np.ndarray


@dataclass(frozen=True)
class AdamVsAdamW:
    adam_final: np.ndarray
    adamw_final: np.ndarray
    adam_param_norm: float
    adamw_param_norm: float
    coordinate_decay_ratio_adam: float
    coordinate_decay_ratio_adamw: float


class _Quadratic(nn.Module):
    """f(w) = 0.5 (w0^2 / a + a w1^2). Minimiser at 0. Condition number a^2."""

    def __init__(self, a: float = 10.0, start: tuple[float, float] = (3.0, 3.0)) -> None:
        super().__init__()
        if a <= 0.0:
            raise ValueError("a must be positive")
        self.a = float(a)
        self.w = nn.Parameter(torch.tensor(start, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        a = self.a
        return 0.5 * (self.w[0] ** 2 / a + a * self.w[1] ** 2)


def _run_opt(opt_name: str, steps: int, lr: float, a: float, start: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    model = _Quadratic(a=a, start=start)
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    else:
        raise ValueError(f"unknown optimiser {opt_name!r}")
    path = [model.w.detach().cpu().numpy().copy()]
    values = [float(model().item())]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        opt.step()
        path.append(model.w.detach().cpu().numpy().copy())
        values.append(float(model().item()))
    return np.stack(path), np.asarray(values, dtype=float)


def run_quadratic_paths(
    steps: int = 40,
    lr_sgd: float = 0.05,
    lr_momentum: float = 0.05,
    lr_adam: float = 0.2,
    a: float = 10.0,
    start: tuple[float, float] = (3.0, 3.0),
) -> QuadraticPaths:
    """Compare paths. SGD and Adam do not share a learning rate.

    Adam's first-moment bias correction makes a shared ``lr=0.05`` look
    like Adam "lost" on this quadratic. That is a step-size artefact, not
    a ranking of methods. Momentum keeps the SGD step size so the extra
    velocity term is the only change.
    """
    sgd, v_sgd = _run_opt("sgd", steps, lr_sgd, a, start)
    mom, _ = _run_opt("momentum", steps, lr_momentum, a, start)
    adam, v_adam = _run_opt("adam", steps, lr_adam, a, start)
    adamw, _ = _run_opt("adamw", steps, lr_adam, a, start)
    return QuadraticPaths(
        sgd=sgd,
        momentum=mom,
        adam=adam,
        adamw=adamw,
        values_sgd=v_sgd,
        values_adam=v_adam,
    )


def run_adam_vs_adamw(
    steps: int = 80,
    lr: float = 0.05,
    weight_decay: float = 0.2,
    a: float = 10.0,
    start: tuple[float, float] = (3.0, 3.0),
) -> AdamVsAdamW:
    """Same λ, same quadratic. Adam couples λ into the adaptive gradient; AdamW does not.

    Coordinate 1 has curvature a times larger than coordinate 0, so Adam's
    second-moment estimate is larger there and coupled L2 is down-weighted
    on that coordinate. AdamW applies a multiplicative decay that does not
    go through 1/sqrt(v).
    """

    def _train(cls: type[torch.optim.Optimizer]) -> np.ndarray:
        model = _Quadratic(a=a, start=start)
        opt = cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = model()
            loss.backward()
            opt.step()
        return model.w.detach().cpu().numpy().copy()

    w_adam = _train(torch.optim.Adam)
    w_adamw = _train(torch.optim.AdamW)
    start_arr = np.asarray(start, dtype=float)
    # How much each coordinate moved toward 0, relative to the other.
    decay_adam = np.abs(w_adam) / np.maximum(np.abs(start_arr), 1e-12)
    decay_adamw = np.abs(w_adamw) / np.maximum(np.abs(start_arr), 1e-12)
    return AdamVsAdamW(
        adam_final=w_adam,
        adamw_final=w_adamw,
        adam_param_norm=float(np.linalg.norm(w_adam)),
        adamw_param_norm=float(np.linalg.norm(w_adamw)),
        coordinate_decay_ratio_adam=float(decay_adam[1] / max(decay_adam[0], 1e-12)),
        coordinate_decay_ratio_adamw=float(decay_adamw[1] / max(decay_adamw[0], 1e-12)),
    )
