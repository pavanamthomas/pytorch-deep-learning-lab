"""Array Dataset and an index split that cannot leak by construction.

What problem is being solved?
    Feed NumPy arrays through ``DataLoader`` and produce train/test index
    sets that are disjoint.

What assumptions are required?
    Rows are iid given the DGP. The split is a random partition of indices,
    not a time-series cut and not a grouped cut.

Why was this method chosen?
    Leakage is a data-handling failure, not an optimiser failure. Making
    the split return two arrays of indices makes the disjointness test a
    one-liner.

What alternative method could have been used?
    sklearn ``train_test_split``; hashing identifiers; a chronological split.

What can go wrong?
    Shuffling after splitting and accidentally concatenating; fitting a
    scaler on train+test; using ``shuffle=True`` on a time-ordered DGP
    (this laboratory does not ship one).

How is correctness independently checked?
    ``tests/test_no_leakage.py`` asserts empty intersection and a partition
    of ``{0, ..., n-1}``.

What can legitimately be concluded?
    The helper's train indices do not meet its test indices.

What cannot be concluded?
    That a caller who bypasses the helper has no leakage, or that random
    splitting is valid for dependent data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from dllab._rng import get_rng


class ArrayPairDataset(Dataset):
    """Indexable pair of feature and label tensors."""

    def __init__(self, x: NDArray, y: NDArray, dtype: torch.dtype = torch.float32) -> None:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x_arr.shape[0] == 0:
            raise ValueError("dataset must be non-empty")
        self.x = torch.as_tensor(x_arr, dtype=dtype)
        if y_arr.dtype.kind in {"i", "u", "b"}:
            self.y = torch.as_tensor(y_arr, dtype=torch.long)
        else:
            self.y = torch.as_tensor(y_arr, dtype=dtype)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class IndexSplit:
    train: NDArray[np.int64]
    test: NDArray[np.int64]


def split_indices(
    n: int,
    test_fraction: float = 0.25,
    seed: int | np.random.Generator | None = 2026,
) -> IndexSplit:
    """Random partition of ``range(n)`` into disjoint train and test index arrays."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    rng = get_rng(seed)
    n_test = int(np.floor(n * test_fraction))
    if n_test < 1 or n_test >= n:
        raise ValueError("test_fraction would leave an empty train or test set")
    perm = rng.permutation(n).astype(np.int64)
    test = perm[:n_test]
    train = perm[n_test:]
    return IndexSplit(train=train, test=test)
