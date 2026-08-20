"""Shared random-number construction.

Every synthetic draw in this package accepts either an integer seed or a
``numpy.random.Generator``. Passing a seed makes the sequence reproducible.
Passing an existing generator lets a caller share state across experiments
without resetting it.

GPU nondeterminism is a separate issue. See ``dllab.experiments.reproducibility``
and ``docs/reproducibility.md``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 2026


def get_rng(seed: int | np.random.Generator | None = DEFAULT_SEED) -> np.random.Generator:
    """Return a NumPy Generator.

    Parameters
    ----------
    seed
        Integer seed, an existing Generator, or ``None`` for draws from the
        OS entropy pool (not reproducible).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
