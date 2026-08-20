# Regularisation

Regularisation here is a change to the finite-sample objective or to the
training procedure. It does not identify a population risk.

## Coupled L2

Adding \(\lambda\|\theta\|^2/2\) to the loss, or equivalently adding
\(\lambda\theta\) to the gradient before SGD, shrinks \(\|\theta\|\)
relative to an unregularised run on the same Gaussian-mixture DGP
(`run_l2_study`). That is a parameter-norm comparison, not a
generalisation theorem.

## Dropout

Inverted dropout zeros coordinates with probability \(p\) at train time
and scales survivors by \(1/(1-p)\) so that the expectation matches
eval, where the map is the identity. Train output is therefore a
different random function of \(x\) than eval output. A caller who
forgets `eval()` measures the noisy map.

## Early stopping

The validation loss is an estimate of risk on a held-out index set from
the *same* DGP, not on a new population. I flip a fraction of training
labels so that a wide MLP can fit noise. The recorded run may then show
validation loss rising after a minimum while training loss is still
falling. When that happens, the correction is to take the checkpoint at
the validation minimum, not to report the last epoch. When it does not
happen, that is also allowed: early stopping is a procedure, not a
guarantee that overfitting is visible in 40 epochs.

## Class weights

Inverse-prevalence weights change the CE risk. They are a regulariser
only in the loose sense that they stop the fit from ignoring a rare
class. See `docs/failure_modes.md`.
