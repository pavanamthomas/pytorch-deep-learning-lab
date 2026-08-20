"""Seed helpers for Python, NumPy, and PyTorch, and a CPU determinism check.

What problem is being solved?
    Make a tiny CPU forward pass repeatable given a documented seed, and
    state what that does *not* buy on a GPU.

What assumptions are required?
    CPU PyTorch with default float32 matmuls is treated as deterministic
    for the tiny graphs in this laboratory. CUDA kernels, cuDNN autotune,
    and some reduction orders are not.

Why was this method chosen?
    A single helper beats three ad hoc ``manual_seed`` calls. The GPU
    caveat belongs in documentation next to the helper, not in a flag
    that silently pretends CUDA is deterministic.

What alternative method could have been used?
    ``torch.use_deterministic_algorithms(True)`` plus
    ``CUBLAS_WORKSPACE_CONFIG``; a context manager that also seeds
    DataLoader workers.

What can go wrong?
    Seeding NumPy's global ``np.random`` but using a ``Generator`` created
    earlier; DataLoader workers with ``num_workers>0`` drawing from
    distinct RNGs; claiming bit-identical GPU training.

How is correctness independently checked?
    Two successive ``seed_everything(2026)`` CPU forwards of a tiny MLP
    match. A second seed produces a different tensor.

What can legitimately be concluded?
    On CPU, this helper repeats the laboratory's tiny graphs.

What cannot be concluded?
    That a CUDA training run is reproducible, even with the same seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from dllab._rng import DEFAULT_SEED


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed ``random``, NumPy's global RNG, and PyTorch (CPU and CUDA if present).

    This does not call ``torch.use_deterministic_algorithms``. On GPU, some
    kernels remain nondeterministic unless extra environment variables are
    set, and even then some operations are disallowed. CI for this repository
    is CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ReproducibilityNote:
    cpu_forwards_match: bool
    cuda_available: bool
    warning: str


GPU_WARNING = (
    "CUDA convolutions, atomic reductions, and cuDNN autotune can disagree "
    "across runs with the same seed. torch.use_deterministic_algorithms(True) "
    "reduces the set of allowed kernels; it is not enabled here because CI is CPU."
)


def run_reproducibility_check(seed: int = DEFAULT_SEED) -> ReproducibilityNote:
    from dllab.torch_net.mlp import TorchMLP

    def _forward() -> torch.Tensor:
        seed_everything(seed)
        model = TorchMLP([3, 4, 2], activation="tanh")
        x = torch.randn(5, 3)
        model.eval()
        with torch.no_grad():
            return model(x)

    a = _forward()
    b = _forward()
    return ReproducibilityNote(
        cpu_forwards_match=bool(torch.equal(a, b)),
        cuda_available=bool(torch.cuda.is_available()),
        warning=GPU_WARNING,
    )
