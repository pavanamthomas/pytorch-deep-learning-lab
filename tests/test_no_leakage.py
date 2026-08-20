"""Train indices produced by the Dataset helper are disjoint from test indices."""

from __future__ import annotations

import numpy as np

from dllab.torch_net.dataset import ArrayPairDataset, split_indices


def test_train_indices_disjoint_from_test() -> None:
    n = 40
    split = split_indices(n, test_fraction=0.25, seed=2026)
    train = set(split.train.tolist())
    test = set(split.test.tolist())
    assert train.isdisjoint(test)
    assert train | test == set(range(n))
    assert len(train) + len(test) == n


def test_array_pair_dataset_length() -> None:
    x = np.zeros((10, 2))
    y = np.arange(10)
    ds = ArrayPairDataset(x, y)
    assert len(ds) == 10
    feat, label = ds[3]
    assert feat.shape == (2,)
    assert int(label.item()) == 3
