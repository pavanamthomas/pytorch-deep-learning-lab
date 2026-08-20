# Roadmap

Current as of August 2026.

## In scope now

- NumPy MLP with affine maps, sigmoid/tanh/ReLU/GELU, MSE and fused softmax CE, SGD.
- Finite-difference gradient checks with an ε sweep; ReLU-at-0 counterexample.
- PyTorch MLP and a tiny 1-D motif CNN; NumPy vs autograd on a shared batch.
- Designed experiments: initialisation, saturation, dead ReLU, SGD/momentum/Adam/AdamW, L2/dropout/early stopping, BatchNorm train vs eval, vanishing/exploding gradients, class-weighted CE, CPU seeding.
- CI: `python -m pytest` and `python scripts/run_all.py`.

## Failures that are part of the design

- Central differences of ReLU at 0 equal 1/2; autograd uses 0.
- Too-large and too-small ε both inflate finite-difference error on a smooth map.
- Zero init ties hidden units; large init saturates sigmoid; deep sigmoid shrinks early gradients.
- BatchNorm train output ≠ eval output on a size-4 probe batch.
- Unweighted CE on a 5%/95% DGP can hide a minority class behind accuracy.

Details: `docs/failures_and_corrections.md`.

## Open (issues)

1. Complex-step derivatives are not implemented; the default check is real central differences.
2. Residual networks, LayerNorm, and learning-rate schedules are out of the current MLP stack.
3. Grouped or temporal splits are not offered; `split_indices` is an iid partition.
4. Early stopping at seed 2026 on the current toy split does not produce a validation rise; the last epoch can be the best epoch. That is allowed, not locked as a required overfit.

## Explicitly not in scope

- Pretrained-model calling, HuggingFace hubs, or ImageNet/GLUE numbers.
- Treating two-moons in-sample accuracy as an empirical finding.
- Claiming GPU bitwise reproducibility.
- Invented latency or production-deployment claims.

Close an issue only with a test or a catalogue/limitation sentence.
