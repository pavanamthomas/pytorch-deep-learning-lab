"""Synthetic data-generating processes used by the laboratory."""

from dllab.data.synthetic import (
    MOTIF,
    ClassificationBatch,
    gaussian_mixture,
    imbalanced_gaussian,
    linearly_separable,
    motif_sequences,
    two_moons,
)

__all__ = [
    "MOTIF",
    "ClassificationBatch",
    "gaussian_mixture",
    "imbalanced_gaussian",
    "linearly_separable",
    "motif_sequences",
    "two_moons",
]
