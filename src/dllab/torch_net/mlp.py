"""PyTorch MLP that mirrors the NumPy stack.

Weight storage is ``(out_features, in_features)``, the ``nn.Linear`` layout.
Copy helpers transpose to and from the NumPy ``(in_features, out_features)``
layout so that a shared initialisation is an identity of the computed map,
not a coincidence of random seeds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dllab.numpy_net.mlp import NumpyMLP


def _activation_module(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {name!r}")


class TorchMLP(nn.Module):
    """Hidden activations; linear last layer. Same depth convention as NumpyMLP."""

    def __init__(self, sizes: list[int], activation: str = "relu") -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least in_features and out_features")
        if any(s < 1 for s in sizes):
            raise ValueError("all sizes must be positive")
        blocks: list[nn.Module] = []
        for i, (in_f, out_f) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            blocks.append(nn.Linear(in_f, out_f))
            if i < len(sizes) - 2:
                blocks.append(_activation_module(activation))
        self.net = nn.Sequential(*blocks)
        self.sizes = list(sizes)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]


def copy_numpy_mlp_to_torch(numpy_net: NumpyMLP, torch_net: TorchMLP) -> None:
    """Overwrite torch parameters with the NumPy weights (transposed)."""
    linear = torch_net.linear_layers()
    if len(linear) != len(numpy_net.layers):
        raise ValueError("depth mismatch between NumPy and Torch MLPs")
    for src, dst in zip(numpy_net.layers, linear, strict=True):
        if tuple(dst.weight.shape) != (src.out_features, src.in_features):
            raise ValueError("layer shapes do not match")
        weight = np.ascontiguousarray(src.weight.T, dtype=np.float64)
        bias = np.ascontiguousarray(src.bias, dtype=np.float64)
        dst.weight.data.copy_(torch.from_numpy(weight))
        dst.bias.data.copy_(torch.from_numpy(bias))
