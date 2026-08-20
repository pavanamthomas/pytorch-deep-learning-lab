"""Manual NumPy neural network: affine maps, activations, losses, SGD.

The public objects are small enough that a derivative can be written on
paper and checked. Nothing here is a pretrained model.
"""

from dllab.numpy_net.activations import (
    apply_activation,
    apply_activation_grad,
    gelu,
    gelu_grad,
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    tanh,
    tanh_grad,
)
from dllab.numpy_net.init import he_normal, xavier_uniform, zeros
from dllab.numpy_net.layers import Affine, Dropout, conv1d_backward, conv1d_forward, conv1d_toeplitz
from dllab.numpy_net.losses import (
    mse,
    mse_grad,
    naive_softmax,
    softmax,
    softmax_cross_entropy,
    softmax_cross_entropy_grad,
)
from dllab.numpy_net.mlp import NumpyMLP, mse_backward_pass, softmax_ce_backward_pass, train_softmax_ce
from dllab.numpy_net.sgd import clip_grad_norm, sgd_step

__all__ = [
    "Affine",
    "Dropout",
    "NumpyMLP",
    "apply_activation",
    "apply_activation_grad",
    "clip_grad_norm",
    "conv1d_backward",
    "conv1d_forward",
    "conv1d_toeplitz",
    "gelu",
    "gelu_grad",
    "he_normal",
    "mse",
    "mse_backward_pass",
    "mse_grad",
    "naive_softmax",
    "relu",
    "relu_grad",
    "sgd_step",
    "sigmoid",
    "sigmoid_grad",
    "softmax",
    "softmax_cross_entropy",
    "softmax_cross_entropy_grad",
    "softmax_ce_backward_pass",
    "tanh",
    "tanh_grad",
    "train_softmax_ce",
    "xavier_uniform",
    "zeros",
]
