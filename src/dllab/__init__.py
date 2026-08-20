"""Neural-network mathematics, a NumPy MLP, and PyTorch autograd checks.

The package is a laboratory: identities are checked on synthetic DGPs.
Nothing here is a pretrained-model wrapper or an empirical leaderboard.
"""

from dllab._rng import DEFAULT_SEED, get_rng
from dllab.experiments.reproducibility import seed_everything
from dllab.numpy_net.mlp import NumpyMLP
from dllab.torch_net.mlp import TorchMLP

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SEED",
    "NumpyMLP",
    "TorchMLP",
    "get_rng",
    "seed_everything",
]
