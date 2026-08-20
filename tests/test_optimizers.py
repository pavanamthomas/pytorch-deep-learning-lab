"""Closed-form SGD on the laboratory quadratic; Adam vs AdamW; shared-lr artefact."""

from __future__ import annotations

import numpy as np
import torch

from dllab.experiments.optimizers import (
    _Quadratic,
    run_adam_vs_adamw,
    run_quadratic_paths,
)


def test_one_sgd_step_matches_closed_form_gradient() -> None:
    a = 10.0
    w0 = torch.tensor([3.0, 3.0], dtype=torch.float64)
    lr = 0.05
    model = _Quadratic(a=a, start=(3.0, 3.0))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.zero_grad(set_to_none=True)
    model().backward()
    opt.step()
    # df/dw = (w0/a, a w1)
    analytic = w0 - lr * torch.tensor([w0[0] / a, a * w0[1]], dtype=torch.float64)
    np.testing.assert_allclose(model.w.detach().cpu().numpy(), analytic.numpy(), atol=1e-12)


def test_adam_and_adamw_diverge_once_weight_decay_is_on() -> None:
    out = run_adam_vs_adamw(steps=80, lr=0.05, weight_decay=0.2)
    assert out.adam_param_norm != out.adamw_param_norm
    # Decoupled decay pulls harder toward the origin on this quadratic.
    assert out.adamw_param_norm < out.adam_param_norm


def test_shared_lr_is_not_a_method_ranking() -> None:
    shared = run_quadratic_paths(steps=40, lr_sgd=0.05, lr_momentum=0.05, lr_adam=0.05)
    split = run_quadratic_paths(steps=40, lr_sgd=0.05, lr_momentum=0.05, lr_adam=0.2)
    f0 = float(shared.values_sgd[0])
    # With a shared small step, Adam can remain above SGD. That is not a ranking.
    assert shared.values_adam[-1] > shared.values_sgd[-1]
    # With a step size chosen for Adam, f falls below the shared-lr Adam path.
    assert split.values_adam[-1] < shared.values_adam[-1]
    assert split.values_adam[-1] < f0
    assert split.values_sgd[-1] < f0
