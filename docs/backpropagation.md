# Backpropagation

The estimand is \(\nabla_\theta L(f_\theta(x), y)\) for a composition of
affine maps and elementwise nonlinearities. The DGP is whatever batch
the caller supplies. Nothing about that batch is identified from the
algebra.

## Affine map

With the NumPy layout \(W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}\),

\[
z = xW + b, \qquad
\frac{\partial L}{\partial W} = x^\top\frac{\partial L}{\partial z}, \qquad
\frac{\partial L}{\partial b} = \sum_n \frac{\partial L}{\partial z_n}, \qquad
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\,W^\top.
\]

PyTorch `nn.Linear` stores the transpose. Comparison code transposes.
Mixing the two layouts is a silent bug, not a statistical one.

## Elementwise activation

If \(h=\sigma(z)\) acts coordinatewise, the incoming gradient is
multiplied by \(\sigma'(z)\). Sigmoid and tanh have \(\sigma'\to 0\)
for large \(|z|\). ReLU has \(\sigma'=0\) on the negative half-line,
including at \(0\) under the autograd convention.

## Fused softmax cross-entropy

A separate softmax Jacobian is \( \mathrm{diag}(p)-pp^\top \). Composed
with CE it simplifies to \(p-y\). The laboratory implements the fused
form. Ill-scaled logits are discussed in `AUTOGRAD_VS_FINITE_DIFFERENCES.md`.

## Convolution as a structured affine map

Valid conv1d with a kernel of length \(K\) is multiplication by a
banded Toeplitz matrix with tied diagonals. The same \(K\) coefficients
are reused at every time index (weight sharing). Each output time depends
on \(K\) inputs (local receptive field). `conv1d_toeplitz` writes that
matrix for a single-channel kernel. That is the conceptual content of
the motif CNN, not a decoration on an MLP.

## What reverse mode does not claim

A correct backward pass is the derivative of the implemented scalar.
It is not evidence that the scalar is a population risk, and it is not
a licence to read causal structure into hidden units.
