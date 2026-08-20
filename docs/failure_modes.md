# Failure modes

These are designed. A test that "fails" the naive procedure is a success
for the laboratory.

## Saturation (sigmoid / tanh)

For \(|z|\) large, \(\sigma'(z)\approx 0\). Deep stacks of sigmoids with
even Xavier-scale weights still shrink \(\|\partial L/\partial W_{\mathrm{first}}\|\)
relative to the last layer (`run_vanishing_study`). Residual connections
and ReLU-family activations are the usual architectural response. They
are not implemented here as a claim of superiority; they are named as
the alternative.

## Dead ReLU

If a unit's pre-activation is negative on the whole batch, its local
derivative is 0 and the incoming weight gradient is 0. A large negative
bias manufactures that. The unit stays dead under SGD. Leaky ReLU is an
alternative; it is out of scope.

## Exploding products

Large initial \(W\) in a deep tanh or linear stack makes the product of
Jacobians huge. `clip_grad_norm_` bounds the step. It does not change
the architecture.

## Zero initialisation

Hidden units remain tied. See `docs/initialization.md`.

## BatchNorm train vs eval

On a batch of size 4, batch mean/variance are noisy. Running statistics
are an EMA of previous batches. The same tensor therefore maps to two
different outputs. Using train mode at evaluation time is a procedure
error. Using eval mode during the first step, before running stats have
seen data, is a different error.

PyTorch also uses biased variance for the training normalisation and an
unbiased update for `running_var`. I record the train/eval gap; I do
not re-derive that estimator choice.

## Unweighted CE on a 5%/95% DGP

Accuracy can be high while minority recall is poor. Inverse-prevalence
weights change the objective. In-sample recall on that DGP is not a
clinical metric.

## Finite differences at a kink or at extreme ε

See `AUTOGRAD_VS_FINITE_DIFFERENCES.md`.

## Leakage

A random index split that is not disjoint is a bug. `split_indices`
returns two arrays that partition `{0,...,n-1}`. Fitting on the union
and then reporting "test" accuracy on a subset is still leakage; the
helper cannot prevent a caller from ignoring it.
