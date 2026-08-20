"""Weight-initialisation schemes for the NumPy affine layers.

What problem is being solved?
    Draw W so that hidden activations neither collapse to a constant nor
    explode at the start of training.

What assumptions are required?
    Units in a layer are iid at initialisation. Fan-in and fan-out are the
    dimensions of that affine map. Biases are zero unless a caller overrides.

Why was this method chosen?
    Zero and large Gaussian draws are the designed failures. Xavier/Glorot
    matches the tanh variance calculation. He matches the ReLU calculation
    (half of the mass is clipped).

What alternative method could have been used?
    Orthogonal initialisation; layer-wise sequential initialisation;
    PyTorch default (Kaiming uniform for ``nn.Linear``).

What can go wrong?
    Xavier on ReLU under-disperses. He on sigmoid still saturates if the
    pre-activation mean is not near 0. Zero init creates a permutation
    symmetry that SGD on a fully connected net does not break.

How is correctness independently checked?
    Experiments measure hidden-unit equality under zero init, saturation
    under large init, and activation variance under Xavier/He on a fixed DGP.

What can legitimately be concluded?
    On the synthetic MLPs in this laboratory, these schemes produce the
    qualitative behaviours documented in ``docs/initialization.md``.

What cannot be concluded?
    That a scheme is optimal for a given architecture, or that initialisation
    alone determines whether training succeeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dllab._rng import get_rng


def zeros(in_features: int, out_features: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if in_features < 1 or out_features < 1:
        raise ValueError("in_features and out_features must be positive")
    return np.zeros((in_features, out_features), dtype=float), np.zeros(out_features, dtype=float)


def gaussian(
    in_features: int,
    out_features: int,
    scale: float,
    seed: int | np.random.Generator | None = 2026,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if in_features < 1 or out_features < 1:
        raise ValueError("in_features and out_features must be positive")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    rng = get_rng(seed)
    w = rng.normal(0.0, scale, size=(in_features, out_features))
    b = np.zeros(out_features, dtype=float)
    return w, b


def xavier_uniform(
    in_features: int,
    out_features: int,
    seed: int | np.random.Generator | None = 2026,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Glorot uniform: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).

    Variance is 2 / (fan_in + fan_out), the tanh-oriented choice.
    """
    if in_features < 1 or out_features < 1:
        raise ValueError("in_features and out_features must be positive")
    rng = get_rng(seed)
    bound = np.sqrt(6.0 / (in_features + out_features))
    w = rng.uniform(-bound, bound, size=(in_features, out_features))
    b = np.zeros(out_features, dtype=float)
    return w, b


def he_normal(
    in_features: int,
    out_features: int,
    seed: int | np.random.Generator | None = 2026,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """He normal: N(0, 2 / fan_in). Intended for ReLU."""
    if in_features < 1 or out_features < 1:
        raise ValueError("in_features and out_features must be positive")
    rng = get_rng(seed)
    std = np.sqrt(2.0 / in_features)
    w = rng.normal(0.0, std, size=(in_features, out_features))
    b = np.zeros(out_features, dtype=float)
    return w, b


def init_affine(
    in_features: int,
    out_features: int,
    scheme: str,
    seed: int | np.random.Generator | None = 2026,
    scale: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    name = scheme.lower()
    if name == "zeros":
        return zeros(in_features, out_features)
    if name in {"gaussian", "large"}:
        return gaussian(in_features, out_features, scale=scale, seed=seed)
    if name in {"xavier", "glorot", "xavier_uniform"}:
        return xavier_uniform(in_features, out_features, seed=seed)
    if name in {"he", "he_normal", "kaiming"}:
        return he_normal(in_features, out_features, seed=seed)
    raise ValueError(f"unknown init scheme {scheme!r}")
