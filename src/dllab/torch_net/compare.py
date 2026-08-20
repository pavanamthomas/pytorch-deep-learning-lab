"""NumPy MLP versus PyTorch autograd on one shared initialisation and one batch.

The estimand is the map (x, θ) -> (logits, dL/dθ) for mean squared error
or mean softmax cross-entropy. Agreement is a numerical identity check,
not a claim that either implementation is a good model of a population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from dllab._rng import get_rng
from dllab.gradcheck.finite_diff import relative_error
from dllab.numpy_net.losses import mse, mse_grad, softmax_cross_entropy, softmax_cross_entropy_grad
from dllab.numpy_net.mlp import NumpyMLP
from dllab.torch_net.mlp import TorchMLP, copy_numpy_mlp_to_torch


@dataclass(frozen=True)
class ForwardBackwardAgreement:
    max_abs_forward_diff: float
    max_abs_loss_diff: float
    max_abs_weight_grad_diff: float
    max_rel_weight_grad_diff: float
    loss_numpy: float
    loss_torch: float


def compare_mse_batch(
    sizes: list[int] | None = None,
    activation: str = "tanh",
    batch: int = 8,
    seed: int | np.random.Generator | None = 2026,
) -> ForwardBackwardAgreement:
    """Compare NumPy reverse mode to autograd on MSE. Tanh avoids the ReLU kink."""
    if sizes is None:
        sizes = [3, 5, 2]
    rng = get_rng(seed)
    x = rng.normal(size=(batch, sizes[0]))
    y = rng.normal(size=(batch, sizes[-1]))
    numpy_net = NumpyMLP.from_sizes(sizes, activation=activation, init="xavier", seed=rng)
    torch_net = TorchMLP(sizes, activation=activation).double()
    copy_numpy_mlp_to_torch(numpy_net, torch_net)

    yhat_np = numpy_net.forward(x)
    loss_np = mse(yhat_np, y)
    numpy_net.backward(mse_grad(yhat_np, y))

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    torch_net.zero_grad(set_to_none=True)
    yhat_t = torch_net(xt)
    loss_t = torch.mean((yhat_t - yt) ** 2)
    loss_t.backward()

    return _summarise(numpy_net, torch_net, yhat_np, yhat_t, loss_np, float(loss_t.item()))


def compare_softmax_ce_batch(
    sizes: list[int] | None = None,
    activation: str = "tanh",
    batch: int = 8,
    seed: int | np.random.Generator | None = 2026,
) -> ForwardBackwardAgreement:
    if sizes is None:
        sizes = [3, 5, 2]
    rng = get_rng(seed)
    x = rng.normal(size=(batch, sizes[0]))
    labels = rng.integers(0, sizes[-1], size=batch)
    numpy_net = NumpyMLP.from_sizes(sizes, activation=activation, init="xavier", seed=rng)
    torch_net = TorchMLP(sizes, activation=activation).double()
    copy_numpy_mlp_to_torch(numpy_net, torch_net)

    logits_np = numpy_net.forward(x)
    loss_np = softmax_cross_entropy(logits_np, labels)
    numpy_net.backward(softmax_cross_entropy_grad(logits_np, labels))

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(labels, dtype=torch.long)
    torch_net.zero_grad(set_to_none=True)
    logits_t = torch_net(xt)
    loss_t = torch.nn.functional.cross_entropy(logits_t, yt)
    loss_t.backward()

    return _summarise(numpy_net, torch_net, logits_np, logits_t, loss_np, float(loss_t.item()))


def _summarise(
    numpy_net: NumpyMLP,
    torch_net: TorchMLP,
    out_np: NDArray[np.floating],
    out_t: torch.Tensor,
    loss_np: float,
    loss_t: float,
) -> ForwardBackwardAgreement:
    out_diff = np.max(np.abs(out_np - out_t.detach().cpu().numpy()))
    grad_diffs: list[float] = []
    rels: list[float] = []
    for src, dst in zip(numpy_net.layers, torch_net.linear_layers(), strict=True):
        if src.grad_weight is None or dst.weight.grad is None:
            raise ValueError("missing weight gradient")
        tw = dst.weight.grad.detach().cpu().numpy().T
        grad_diffs.append(float(np.max(np.abs(src.grad_weight - tw))))
        rels.append(relative_error(src.grad_weight, tw))
        if src.grad_bias is None or dst.bias.grad is None:
            raise ValueError("missing bias gradient")
        tb = dst.bias.grad.detach().cpu().numpy()
        grad_diffs.append(float(np.max(np.abs(src.grad_bias - tb))))
        rels.append(relative_error(src.grad_bias, tb))
    return ForwardBackwardAgreement(
        max_abs_forward_diff=float(out_diff),
        max_abs_loss_diff=abs(loss_np - loss_t),
        max_abs_weight_grad_diff=max(grad_diffs),
        max_rel_weight_grad_diff=max(rels),
        loss_numpy=loss_np,
        loss_torch=loss_t,
    )
