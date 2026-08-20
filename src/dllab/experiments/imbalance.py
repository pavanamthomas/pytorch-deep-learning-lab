"""Weighted versus unweighted cross-entropy on a 5%/95% Gaussian DGP.

The estimand is minority-class recall after a short full-batch fit of a
linear (logistic) head. The DGP is ``imbalanced_gaussian``: class 1 has a
mean shift on the first coordinate and prevalence 0.05 by default.

Unweighted CE minimises a risk that is almost entirely majority-class
error. A class-weight vector proportional to inverse prevalence changes
that risk. Neither fit is a claim about a medical or credit dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dllab.data.synthetic import imbalanced_gaussian
from dllab.numpy_net.mlp import NumpyMLP, train_softmax_ce


@dataclass(frozen=True)
class ImbalanceStudy:
    prevalence: float
    unweighted_recall: float
    weighted_recall: float
    unweighted_accuracy: float
    weighted_accuracy: float
    n: int


def _metrics(logits: NDArray[np.floating], y: NDArray[np.integer]) -> tuple[float, float]:
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    minority = y == 1
    if not np.any(minority):
        raise ValueError("DGP produced no minority labels")
    recall = float(np.mean(pred[minority] == 1))
    return acc, recall


def inverse_prevalence_weights(y: NDArray[np.integer], n_classes: int = 2) -> NDArray[np.floating]:
    """w_c = n / (n_classes * n_c), the 'balanced' heuristic. Zero counts are rejected."""
    y = np.asarray(y, dtype=int).reshape(-1)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("every class must appear if inverse-prevalence weights are used")
    n = float(y.size)
    return n / (float(n_classes) * counts)


def run_imbalance_study(
    n: int = 240,
    prevalence: float = 0.05,
    epochs: int = 40,
    lr: float = 0.4,
    seed: int = 2026,
) -> ImbalanceStudy:
    data = imbalanced_gaussian(n=n, prevalence=prevalence, seed=seed)
    weights = inverse_prevalence_weights(data.y, n_classes=2)

    unweighted = NumpyMLP.from_sizes([2, 2], activation="relu", init="xavier", seed=seed)
    # A 2->2 linear model: from_sizes([2, 2]) has no hidden activation, which is logistic CE.
    train_softmax_ce(unweighted, data.x, data.y, lr=lr, epochs=epochs)

    weighted = NumpyMLP.from_sizes([2, 2], activation="relu", init="xavier", seed=seed)
    train_softmax_ce(weighted, data.x, data.y, lr=lr, epochs=epochs, class_weight=weights)

    acc_u, rec_u = _metrics(unweighted.forward(data.x), data.y)
    acc_w, rec_w = _metrics(weighted.forward(data.x), data.y)
    return ImbalanceStudy(
        prevalence=float(np.mean(data.y == 1)),
        unweighted_recall=rec_u,
        weighted_recall=rec_w,
        unweighted_accuracy=acc_u,
        weighted_accuracy=acc_w,
        n=n,
    )
