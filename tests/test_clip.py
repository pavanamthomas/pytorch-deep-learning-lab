"""Global L2 gradient clipping respects the bound."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dllab.experiments.gradients import run_exploding_and_clip, torch_clip_example
from dllab.numpy_net.sgd import clip_grad_norm


def test_numpy_clip_bound() -> None:
    g1 = np.array([3.0, 4.0])
    g2 = np.array([0.0])
    grads = [g1.copy(), g2.copy()]
    before = clip_grad_norm(grads, max_norm=1.0)
    assert abs(before - 5.0) < 1e-12
    total = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))
    assert total <= 1.0 + 1e-12
    np.testing.assert_allclose(grads[0], np.array([0.6, 0.8]))


def test_torch_clip_bound() -> None:
    before, after = torch_clip_example(max_norm=1.0)
    assert abs(before - 5.0) < 1e-12
    # PyTorch divides by (norm + 1e-6), so the clipped norm is slightly under max_norm.
    assert after <= 1.0 + 1e-12
    assert abs(after - 1.0) < 1e-5
    param = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    param.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
    assert torch.allclose(param.grad, torch.tensor([0.6, 0.8], dtype=torch.float64), atol=1e-5, rtol=0.0)


def test_exploding_then_clip_respects_bound() -> None:
    study = run_exploding_and_clip(seed=2026, max_norm=1.0)
    assert study.unclipped_norm > 1.0
    assert study.respected_bound
    assert study.clipped_norm <= study.max_norm + 1e-9
