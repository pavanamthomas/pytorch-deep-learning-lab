"""L2, dropout, and early stopping on tiny synthetic classification problems.

None of these is a claim about a production regulariser. Early stopping
needs a validation split; that split is an index partition, not a second
population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dllab.data.synthetic import gaussian_mixture
from dllab.numpy_net.layers import Dropout
from dllab.numpy_net.mlp import NumpyMLP, softmax_ce_backward_pass
from dllab.torch_net.dataset import split_indices
from dllab.torch_net.mlp import TorchMLP


@dataclass(frozen=True)
class L2Study:
    unregularised_weight_norm: float
    l2_weight_norm: float


@dataclass(frozen=True)
class DropoutStudy:
    train_output_mean_abs: float
    eval_output_mean_abs: float
    train_frac_exact_zero: float


@dataclass(frozen=True)
class EarlyStoppingStudy:
    best_epoch: int
    train_loss_at_best: float
    val_loss_at_best: float
    train_loss_final: float
    val_loss_final: float
    val_rose_after_best: bool


def run_l2_study(
    n: int = 40,
    epochs: int = 25,
    lr: float = 0.2,
    weight_decay: float = 0.15,
    seed: int = 2026,
) -> L2Study:
    data = gaussian_mixture(n=n, seed=seed)
    net_free = NumpyMLP.from_sizes([2, 8, 2], activation="tanh", init="xavier", seed=seed)
    net_l2 = NumpyMLP.from_sizes([2, 8, 2], activation="tanh", init="xavier", seed=seed)
    for _ in range(epochs):
        softmax_ce_backward_pass(net_free, data.x, data.y)
        net_free.step(lr=lr, weight_decay=0.0)
        softmax_ce_backward_pass(net_l2, data.x, data.y)
        net_l2.step(lr=lr, weight_decay=weight_decay)

    def _norm(net: NumpyMLP) -> float:
        return float(np.sqrt(sum(float(np.sum(p**2)) for p in net.parameters())))

    return L2Study(
        unregularised_weight_norm=_norm(net_free),
        l2_weight_norm=_norm(net_l2),
    )


def run_dropout_study(
    n: int = 32,
    p: float = 0.5,
    seed: int = 2026,
) -> DropoutStudy:
    rng_data = gaussian_mixture(n=n, seed=seed)
    drop = Dropout(p=p)
    train_out = drop.forward(rng_data.x, train=True, seed=seed)
    eval_out = drop.forward(rng_data.x, train=False, seed=seed)
    return DropoutStudy(
        train_output_mean_abs=float(np.mean(np.abs(train_out))),
        eval_output_mean_abs=float(np.mean(np.abs(eval_out))),
        train_frac_exact_zero=float(np.mean(train_out == 0.0)),
    )


def run_early_stopping_study(
    n: int = 48,
    hidden: int = 24,
    epochs: int = 40,
    lr: float = 0.08,
    seed: int = 2026,
) -> EarlyStoppingStudy:
    """Wide MLP on a small noisy mixture. Train loss can keep falling after val rises."""
    data = gaussian_mixture(n=n, seed=seed)
    split = split_indices(n, test_fraction=0.4, seed=seed)
    x_tr, y_tr = data.x[split.train], data.y[split.train]
    x_va, y_va = data.x[split.test], data.y[split.test]
    # Label noise on the training side only, so the fit can overfit the noise.
    rng = np.random.default_rng(seed + 1)
    flip = rng.random(y_tr.shape[0]) < 0.25
    y_tr_noisy = y_tr.copy()
    y_tr_noisy[flip] = 1 - y_tr_noisy[flip]

    model = TorchMLP([2, hidden, hidden, 2], activation="relu")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_hist: list[float] = []
    val_hist: list[float] = []

    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr_noisy, dtype=torch.long)
    xv = torch.tensor(x_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        train_hist.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_hist.append(float(loss_fn(model(xv), yv).item()))

    best = int(np.argmin(val_hist))
    return EarlyStoppingStudy(
        best_epoch=best,
        train_loss_at_best=train_hist[best],
        val_loss_at_best=val_hist[best],
        train_loss_final=train_hist[-1],
        val_loss_final=val_hist[-1],
        val_rose_after_best=bool(val_hist[-1] > val_hist[best] + 1e-12),
    )
