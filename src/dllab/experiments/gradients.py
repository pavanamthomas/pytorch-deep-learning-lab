"""Vanishing gradients in a deep sigmoid stack, exploding products, and clip_grad_norm_.

The estimand is the Euclidean norm of dL/dW for the *first* affine map,
relative to the last, on a fixed batch. The DGP is Gaussian x and random
labels; it is a probe of the architecture, not a classification task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dllab._rng import get_rng
from dllab.numpy_net.mlp import NumpyMLP, softmax_ce_backward_pass
from dllab.numpy_net.sgd import clip_grad_norm


@dataclass(frozen=True)
class VanishingStudy:
    depth: int
    first_layer_grad_norm: float
    last_layer_grad_norm: float
    ratio_first_to_last: float


@dataclass(frozen=True)
class ExplodingStudy:
    unclipped_norm: float
    clipped_norm: float
    max_norm: float
    respected_bound: bool


def run_vanishing_study(
    depth: int = 8,
    width: int = 6,
    n: int = 16,
    seed: int | np.random.Generator | None = 2026,
) -> VanishingStudy:
    """Deep sigmoid MLP with Xavier-scale weights. First-layer grads shrink."""
    if depth < 3:
        raise ValueError("depth must be at least 3")
    rng = get_rng(seed)
    sizes = [width] * (depth + 1)
    sizes[0] = 4
    sizes[-1] = 2
    net = NumpyMLP.from_sizes(sizes, activation="sigmoid", init="xavier", seed=rng)
    x = rng.normal(size=(n, sizes[0]))
    y = rng.integers(0, 2, size=n)
    softmax_ce_backward_pass(net, x, y)
    g_first = net.layers[0].grad_weight
    g_last = net.layers[-1].grad_weight
    assert g_first is not None and g_last is not None
    n_first = float(np.linalg.norm(g_first))
    n_last = float(np.linalg.norm(g_last))
    return VanishingStudy(
        depth=depth,
        first_layer_grad_norm=n_first,
        last_layer_grad_norm=n_last,
        ratio_first_to_last=n_first / max(n_last, 1e-18),
    )


def run_exploding_and_clip(
    depth: int = 6,
    width: int = 6,
    n: int = 12,
    large_scale: float = 3.5,
    max_norm: float = 1.0,
    seed: int | np.random.Generator | None = 2026,
) -> ExplodingStudy:
    rng = get_rng(seed)
    sizes = [width] * (depth + 1)
    sizes[0] = 4
    sizes[-1] = 2
    net = NumpyMLP.from_sizes(
        sizes, activation="tanh", init="gaussian", seed=rng, scale=large_scale
    )
    x = rng.normal(size=(n, sizes[0]))
    y = rng.integers(0, 2, size=n)
    softmax_ce_backward_pass(net, x, y)
    grads = [g.copy() for g in net.grads()]
    unclipped = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))
    clipped_copy = [g.copy() for g in grads]
    clip_grad_norm(clipped_copy, max_norm=max_norm)
    clipped = float(np.sqrt(sum(float(np.sum(g**2)) for g in clipped_copy)))
    return ExplodingStudy(
        unclipped_norm=unclipped,
        clipped_norm=clipped,
        max_norm=max_norm,
        respected_bound=bool(clipped <= max_norm + 1e-9),
    )


def torch_clip_example(max_norm: float = 1.0) -> tuple[float, float]:
    """Vector (3, 4) has norm 5. After clip to 1, norm is 1 and the vector is (0.6, 0.8)."""
    param = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    param.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    before = float(param.grad.norm().item())
    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm)
    after = float(param.grad.norm().item())
    return before, after
