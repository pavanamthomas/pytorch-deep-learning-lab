"""SGD on explicit parameter arrays. No momentum, no autograd.

What problem is being solved?
    Take a list of parameters and a matching list of gradients and apply
    w <- w - lr * (g + weight_decay * w).

What assumptions are required?
    Gradients are already the derivatives of the scalar being minimised,
    including any batch averaging. The optional L2 term is coupled into the
    gradient (the classical SGD-with-weight-decay form), not AdamW-style
    decoupling.

Why was this method chosen?
    It is the update written in textbooks next to backpropagation. Matching
    ``torch.optim.SGD`` with momentum=0 is then a one-line check.

What alternative method could have been used?
    Momentum, Nesterov, Adam, L-BFGS.

What can go wrong?
    Coupled L2 is not the same object as decoupled weight decay once the
    optimiser is adaptive. A learning rate that is large relative to the
    curvature diverges.

How is correctness independently checked?
    A linearly separable toy set: mean softmax CE decreases after several
    steps. A quadratic with a known gradient matches a hand-written step.

What can legitimately be concluded?
    This update is steepest descent on the current minibatch objective,
    plus optional coupled L2.

What cannot be concluded?
    That SGD finds a global minimiser, or that training loss decrease
    implies a small generalisation gap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sgd_step(
    params: list[NDArray[np.floating]],
    grads: list[NDArray[np.floating]],
    lr: float,
    weight_decay: float = 0.0,
) -> None:
    """In-place SGD. ``weight_decay`` is coupled L2: g <- g + wd * p."""
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if len(params) != len(grads):
        raise ValueError("params and grads must have the same length")
    for p, g in zip(params, grads, strict=True):
        if p.shape != g.shape:
            raise ValueError(f"param shape {p.shape} does not match grad shape {g.shape}")
        update = np.asarray(g, dtype=float)
        if weight_decay != 0.0:
            update = update + weight_decay * p
        p -= lr * update


def clip_grad_norm(
    grads: list[NDArray[np.floating]],
    max_norm: float,
    eps: float = 1e-12,
) -> float:
    """Scale all grads so that the concatenated Euclidean norm is at most ``max_norm``.

    Returns the global norm before clipping. This is the same contract as
    ``torch.nn.utils.clip_grad_norm_`` with ``norm_type=2``.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    total_sq = 0.0
    for g in grads:
        total_sq += float(np.sum(np.asarray(g, dtype=float) ** 2))
    total_norm = float(np.sqrt(total_sq))
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g *= clip_coef
    return total_norm
