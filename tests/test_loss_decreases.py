"""Softmax CE decreases on a linearly separable toy DGP."""

from __future__ import annotations

from dllab.data.synthetic import linearly_separable
from dllab.numpy_net.mlp import NumpyMLP, train_softmax_ce


def test_loss_decreases_on_linearly_separable_toy() -> None:
    data = linearly_separable(n=40, seed=2026)
    net = NumpyMLP.from_sizes([2, 8, 2], activation="relu", init="he", seed=2026)
    hist = train_softmax_ce(net, data.x, data.y, lr=0.4, epochs=30)
    assert hist[-1] < hist[0]
    assert hist[-1] < 0.4
