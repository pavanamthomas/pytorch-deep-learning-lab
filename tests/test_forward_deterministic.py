"""A tiny forward pass is a deterministic function of the seed."""

from __future__ import annotations

import numpy as np
import torch

from dllab.experiments.reproducibility import run_reproducibility_check, seed_everything
from dllab.numpy_net.mlp import NumpyMLP
from dllab.torch_net.mlp import TorchMLP


def test_numpy_forward_deterministic() -> None:
    x = np.ones((3, 2))
    a = NumpyMLP.from_sizes([2, 4, 2], activation="tanh", seed=2026).forward(x)
    b = NumpyMLP.from_sizes([2, 4, 2], activation="tanh", seed=2026).forward(x)
    assert np.array_equal(a, b)


def test_torch_cpu_forward_deterministic() -> None:
    note = run_reproducibility_check(seed=2026)
    assert note.cpu_forwards_match


def test_second_seed_changes_torch_forward() -> None:
    def _fwd(seed: int) -> torch.Tensor:
        seed_everything(seed)
        model = TorchMLP([3, 4, 2], activation="tanh")
        x = torch.randn(5, 3)
        with torch.no_grad():
            return model(x)

    assert not torch.equal(_fwd(2026), _fwd(2027))
