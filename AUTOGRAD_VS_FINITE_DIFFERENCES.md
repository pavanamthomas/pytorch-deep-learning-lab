# Manual derivatives, finite differences, and autograd

This note is the flagship check of the laboratory. The estimand is the
gradient of a scalar training loss with respect to an explicit parameter
array. Three procedures compute it:

1. a reverse-mode formula written from the chain rule,
2. a central finite-difference quotient,
3. PyTorch autograd.

Agreement of (1) and (3) on a smooth map is evidence that the NumPy
backward pass matches the scalar that autograd differentiates. Agreement
of (1) and (2) at a well-chosen ε is evidence that the scalar is being
differentiated at all. Disagreement is not automatically a bug: ε can be
wrong, and the map can fail to be differentiable.

The numbers for one seed are printed by `python scripts/run_all.py`.
They are not copied here, so they cannot rot.

## 1. Manual reverse mode

Take one affine map without bias, a tanh, and mean squared error.

\[
z = XW, \qquad \hat y = \tanh(z), \qquad
L = \frac{1}{N}\sum_{n,c}(\hat y_{nc} - y_{nc})^2
\]

with \(N = n_{\text{batch}}\cdot n_{\text{out}}\). Then

\[
\frac{\partial L}{\partial \hat y} = \frac{2}{N}(\hat y - y), \qquad
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat y}\odot (1-\tanh(z)^2), \qquad
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial z}.
\]

That is the analytic gradient implemented in
`dllab.gradcheck.three_way` and, for a stacked MLP, in `dllab.numpy_net`.

ReLU replaces \(1-\tanh(z)^2\) by the indicator \(1_{z>0}\), with the
conventional value \(0\) at \(z=0\). Softmax cross-entropy is fused:
if \(p = \mathrm{softmax}(z)\) and \(y\) is one-hot,

\[
\frac{\partial L}{\partial z} = \frac{1}{n_{\text{batch}}}(p-y)
\]

for a mean reduction. Differentiating softmax and CE separately is
possible and more poorly scaled.

## 2. Central finite differences

For a coordinate \(\theta_i\),

\[
\widehat{\partial_i L}
= \frac{L(\theta+\varepsilon e_i) - L(\theta-\varepsilon e_i)}{2\varepsilon}.
\]

Taylor expansion with a remainder gives truncation error \(O(\varepsilon^2)\)
when \(L\) is \(C^3\). The numerator is a subtraction of nearly equal
floating-point numbers when \(\varepsilon\) is small, so the cancellation
error is on the order of \(\varepsilon_{\mathrm{mach}}/\varepsilon\) times
a Lipschitz factor of \(L\).

Those two terms trade off. Too large \(\varepsilon\): the \(O(\varepsilon^2)\)
remainder dominates. Too small \(\varepsilon\): digits of \(L(\theta+\varepsilon)\)
and \(L(\theta-\varepsilon)\) agree, and the quotient is noise. The sweep
in `dllab.gradcheck.finite_diff.sweep_epsilon` is the picture of that
trade-off on a tanh+MSE map in float64. A typical good ε on that map is
near \(10^{-5}\) or \(10^{-6}\). The ends of the sweep are designed to
look worse.

Forward differences have truncation \(O(\varepsilon)\) and are not used
as the default check.

## 3. Autograd

PyTorch records elementary ops on a tensor graph and applies the same
chain rule. For the affine+tanh+MSE map, `nn.Linear` stores \(W\) as
`(out, in)`. The laboratory transposes when copying from NumPy, so the
mathematical map is the same object. `tests/test_numpy_torch_compare.py`
checks a full MLP, not only the one-layer flagship.

Autograd is not exact arithmetic. It is reverse mode in finite precision,
with implementation conventions at kinks.

## 4. A case that is numerically unstable: ReLU at 0

\(\mathrm{relu}(t)=\max(t,0)\) is not differentiable at \(0\). The
subdifferential is \([0,1]\). This laboratory and PyTorch take the value
\(0\).

Central differences do something else. For any \(\varepsilon>0\) that
does not overflow,

\[
\frac{\mathrm{relu}(\varepsilon)-\mathrm{relu}(-\varepsilon)}{2\varepsilon}
= \frac{\varepsilon-0}{2\varepsilon}=\frac12.
\]

The forward difference is \(1\). The backward difference is \(0\). The
gap between central differences and the autograd convention **does not
go to zero with \(\varepsilon\)**. That is not truncation versus
cancellation. It is a point at which the finite-difference hypothesis
("\(L\) is \(C^2\) nearby") is false.

The check `relu_kink_central_difference` records this on purpose.

Away from \(0\), ReLU is linear on each side and finite differences
recover the analytic derivative exactly in exact arithmetic.

## 5. A case that is ill-scaled: softmax

Naive `exp(z)/sum(exp(z))` overflows for large logits. The laboratory
exposes that as `naive_softmax`. The production path uses a max-shift
and fused log-softmax.

Even with a stable implementation, logits on the order of \(10^3\) make
softmax a numerical one-hot. Coordinates that have already underflowed
to zero have analytic Jacobian entries of zero, and finite differences
of those coordinates are then a ratio of zeros or of underflow residuals.
A "gradient check" in that region can look like agreement (both sides
~0) without having tested the Jacobian of a non-degenerate softmax.

The right response is to check softmax-CE on well-scaled logits, which
is what `compare_softmax_ce_batch` does, and to treat huge logits as a
saturation diagnostic, not as a precision test.

## What this does not show

- That a training run on two-moons has a small generalisation gap.
- That autograd is correct for every kernel PyTorch ships, including
  CUDA reductions that are allowed to be nondeterministic.
- That a finite-difference failure at extreme ε is a bug in the backward
  pass.

The independent checks are the tests named in `docs/failures_and_corrections.md`.
