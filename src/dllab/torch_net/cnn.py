"""Tiny 1-D CNN whose inductive bias is a local receptive field plus weight sharing.

A kernel of length 3 applied at every time index is a different hypothesis
class from a dense map of the flattened sequence. The motif DGP is built so
that this restriction is relevant: the signal is a local pattern, not a
global linear trend.
"""

from __future__ import annotations

from torch import nn


class MotifCNN(nn.Module):
    """Conv1d(1 -> n_filters, k=3) -> ReLU -> global max pool -> linear logits."""

    def __init__(self, n_filters: int = 4, kernel_size: int = 3, n_classes: int = 2) -> None:
        super().__init__()
        if n_filters < 1 or kernel_size < 1 or n_classes < 2:
            raise ValueError("n_filters, kernel_size must be positive; n_classes >= 2")
        self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.pool(h).squeeze(-1)
        return self.head(h)
