"""Affine layers, dropout, and a 1-D convolution written as explicit maps.

What problem is being solved?
    Represent the linear pieces of a network so that forward values and
    reverse-mode gradients can be computed by hand, without autograd.

What assumptions are required?
    Affine: ``y = x @ W + b`` with ``W`` of shape ``(in_features, out_features)``.
    That is the NumPy layout. PyTorch ``nn.Linear`` stores ``weight`` as
    ``(out_features, in_features)``; comparison code transposes.
    Conv1d: valid (no padding) correlation along the last axis.
    Dropout: inverted dropout, mask drawn only in training.

Why was this method chosen?
    An explicit affine map makes dW = x.T @ dy visible. An explicit conv
    makes weight sharing visible: one kernel coefficient receives a sum of
    local contributions.

What alternative method could have been used?
    Einstein-summation kernels; im2col; storing weights already transposed
    to match PyTorch.

What can go wrong?
    Mixing (in, out) with (out, in) silently transposes the model.
    Dropout without inverted scaling changes the expected activation at
    evaluation. Valid convolution shortens the sequence; shapes then fail.

How is correctness independently checked?
    Affine gradients against a closed-form affine+ReLU+MSE test.
    Conv forward against the equivalent banded Toeplitz matrix.
    NumPy vs PyTorch on a shared initialisation.

What can legitimately be concluded?
    These maps and their reverse-mode formulae are implemented as stated.

What cannot be concluded?
    That a convolution is the right inductive bias for an empirical series,
    or that dropout is a calibrated Bayesian approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dllab._rng import get_rng
from dllab.numpy_net.activations import apply_activation, apply_activation_grad


@dataclass
class Affine:
    """y = x @ W + b, with W shaped (in_features, out_features)."""

    weight: NDArray[np.floating]
    bias: NDArray[np.floating]
    _input: NDArray[np.floating] | None = field(default=None, init=False, repr=False)
    grad_weight: NDArray[np.floating] | None = field(default=None, init=False, repr=False)
    grad_bias: NDArray[np.floating] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.weight, dtype=float)
        b = np.asarray(self.bias, dtype=float)
        if w.ndim != 2:
            raise ValueError("weight must have shape (in_features, out_features)")
        if b.shape != (w.shape[1],):
            raise ValueError("bias must have shape (out_features,)")
        self.weight = w
        self.bias = b

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[1])

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError("affine input must have shape (batch, in_features)")
        if arr.shape[1] != self.in_features:
            raise ValueError(
                f"expected in_features={self.in_features}, got {arr.shape[1]}"
            )
        self._input = arr
        return arr @ self.weight + self.bias

    def backward(self, grad_output: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._input is None:
            raise ValueError("backward called before forward")
        dy = np.asarray(grad_output, dtype=float)
        if dy.shape != (self._input.shape[0], self.out_features):
            raise ValueError(
                f"grad_output shape {dy.shape} does not match "
                f"{(self._input.shape[0], self.out_features)}"
            )
        self.grad_weight = self._input.T @ dy
        self.grad_bias = dy.sum(axis=0)
        return dy @ self.weight.T


@dataclass
class Dropout:
    """Inverted dropout. At train time, keep with probability 1-p and scale by 1/(1-p)."""

    p: float = 0.5
    _mask: NDArray[np.floating] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p < 1.0:
            raise ValueError("dropout p must satisfy 0 <= p < 1")

    def forward(
        self,
        x: NDArray[np.floating],
        *,
        train: bool,
        seed: int | np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        arr = np.asarray(x, dtype=float)
        if not train or self.p == 0.0:
            self._mask = np.ones_like(arr)
            return arr
        rng = get_rng(seed)
        keep = 1.0 - self.p
        mask = (rng.random(arr.shape) < keep).astype(float) / keep
        self._mask = mask
        return arr * mask

    def backward(self, grad_output: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._mask is None:
            raise ValueError("backward called before forward")
        return np.asarray(grad_output, dtype=float) * self._mask


def conv1d_forward(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    bias: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Valid conv1d: x (N, C_in, L), weight (C_out, C_in, K) -> (N, C_out, L-K+1).

    This is cross-correlation, matching ``torch.nn.Conv1d`` (no kernel flip).
    """
    arr = np.asarray(x, dtype=float)
    w = np.asarray(weight, dtype=float)
    if arr.ndim != 3 or w.ndim != 3:
        raise ValueError("x must be (N, C_in, L) and weight (C_out, C_in, K)")
    n, c_in, length = arr.shape
    c_out, c_in_w, kernel = w.shape
    if c_in != c_in_w:
        raise ValueError("in_channels of x and weight differ")
    if kernel > length:
        raise ValueError("kernel larger than sequence length")
    out_len = length - kernel + 1
    out = np.zeros((n, c_out, out_len), dtype=float)
    for t in range(out_len):
        window = arr[:, :, t : t + kernel]
        # window (N, C_in, K), weight (C_out, C_in, K)
        out[:, :, t] = np.tensordot(window, w, axes=([1, 2], [1, 2]))
    if bias is not None:
        b = np.asarray(bias, dtype=float).reshape(-1)
        if b.shape != (c_out,):
            raise ValueError("bias must have shape (C_out,)")
        out += b[None, :, None]
    return out


def conv1d_backward(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    grad_output: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return (dx, dweight, dbias) for valid conv1d.

    Weight sharing is the sum over time: dW[o, i, k] = sum_{n,t} dy[n,o,t] x[n,i,t+k].
    """
    arr = np.asarray(x, dtype=float)
    w = np.asarray(weight, dtype=float)
    dy = np.asarray(grad_output, dtype=float)
    n, c_in, length = arr.shape
    c_out, _, kernel = w.shape
    out_len = length - kernel + 1
    if dy.shape != (n, c_out, out_len):
        raise ValueError(f"grad_output shape {dy.shape} does not match {(n, c_out, out_len)}")
    d_w = np.zeros_like(w, dtype=float)
    d_x = np.zeros_like(arr, dtype=float)
    d_b = dy.sum(axis=(0, 2))
    for t in range(out_len):
        window = arr[:, :, t : t + kernel]
        # dW += einsum('not,nik->oik', dy[:,:,t], window)
        d_w += np.tensordot(dy[:, :, t], window, axes=([0], [0]))
        # dx window += einsum('not,oik->nik', dy[:,:,t], w)
        d_x[:, :, t : t + kernel] += np.tensordot(dy[:, :, t], w, axes=([1], [0]))
    return d_x, d_w, d_b


def conv1d_toeplitz(weight: NDArray[np.floating], length: int) -> NDArray[np.floating]:
    """Dense matrix equivalent of a single-batch, single-in, single-out conv1d.

    For weight shape (1, 1, K) the map x flattened (L,) -> out flattened (L-K+1,)
    is multiplication by a banded Toeplitz matrix with tied diagonals. That
    matrix is the definition of weight sharing plus a local receptive field.
    """
    w = np.asarray(weight, dtype=float)
    if w.shape[0] != 1 or w.shape[1] != 1:
        raise ValueError("toeplitz helper expects weight shape (1, 1, K)")
    kernel = w.shape[2]
    if kernel > length:
        raise ValueError("kernel larger than sequence length")
    out_len = length - kernel + 1
    mat = np.zeros((out_len, length), dtype=float)
    coeff = w[0, 0]
    for t in range(out_len):
        mat[t, t : t + kernel] = coeff
    return mat


class Activation:
    """Named elementwise activation with a cached pre-activation for backward."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._pre: NDArray[np.floating] | None = None

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        arr = np.asarray(x, dtype=float)
        self._pre = arr
        return apply_activation(self.name, arr)

    def backward(self, grad_output: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._pre is None:
            raise ValueError("backward called before forward")
        return np.asarray(grad_output, dtype=float) * apply_activation_grad(self.name, self._pre)
