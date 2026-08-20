"""BatchNorm train versus eval: batch statistics versus running statistics.

What problem is being solved?
    Make the train/eval discrepancy of BatchNorm visible on a small batch.

What assumptions are required?
    ``nn.BatchNorm1d`` in training uses the current batch mean and the
    *biased* batch variance, and updates running estimates with momentum.
    In eval it uses those running estimates. The running variance uses an
    unbiased update. That is PyTorch's contract, not a universal definition.

Why was this method chosen?
    A batch of size 4 after a few noisy updates makes running mean and
    batch mean disagree. That disagreement is the object of interest.

What alternative method could have been used?
    LayerNorm (no running stats); GroupNorm; disabling
    ``track_running_stats``.

What can go wrong?
    Evaluating in train mode on a tiny batch; deploying a model that was
    never switched to eval; interpreting the discrepancy as a bug.

How is correctness independently checked?
    ``tests/test_batchnorm.py`` asserts that after training, the same
    small batch yields different outputs in train and eval modes.

What can legitimately be concluded?
    On this batch, train-mode output is not eval-mode output.

What cannot be concluded?
    That BatchNorm is required for the two-moons MLP, or that running
    statistics have converged to a population moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BatchNormStudy:
    max_abs_train_eval_diff: float
    batch_mean: float
    running_mean: float
    batch_size: int


class _BNNet(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


def run_batchnorm_study(
    dim: int = 4,
    n_train: int = 64,
    batch_size: int = 4,
    steps: int = 15,
    seed: int = 2026,
) -> BatchNormStudy:
    torch.manual_seed(seed)
    model = _BNNet(dim=dim)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # Population-like stream the running stats see during training.
    for i in range(steps):
        model.train()
        x = torch.randn(n_train, dim) + 0.4 * float(i % 3)
        opt.zero_grad(set_to_none=True)
        y = model(x)
        # Dummy loss so gamma/beta move a little; running stats update regardless.
        loss = y.pow(2).mean()
        loss.backward()
        opt.step()

    probe = torch.randn(batch_size, dim) * 2.5 + 1.7
    model.train()
    train_out = model(probe).detach()
    batch_mean = float(probe.mean())
    model.eval()
    eval_out = model(probe).detach()
    running_mean = float(model.bn.running_mean.mean().item())  # type: ignore[union-attr]
    diff = float((train_out - eval_out).abs().max().item())
    return BatchNormStudy(
        max_abs_train_eval_diff=diff,
        batch_mean=batch_mean,
        running_mean=running_mean,
        batch_size=batch_size,
    )
