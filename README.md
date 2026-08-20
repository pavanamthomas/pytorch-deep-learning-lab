# pytorch-deep-learning-lab

[![CI](https://github.com/pavanamthomas/pytorch-deep-learning-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/pytorch-deep-learning-lab/actions)

Neural-network mathematics, a transparent NumPy multilayer perceptron,
and PyTorch autograd checks on synthetic data-generating processes.

This repository is a laboratory. It implements affine maps, activations,
losses, and SGD by hand, checks those derivatives against finite
differences and against autograd, and records designed failures
(saturation, dead ReLU, vanishing and exploding gradients, BatchNorm
train/eval mismatch, unweighted CE on a rare class). It does not wrap
pretrained models and it does not report a leaderboard.

Author: Dr. Pavanam Thomas ([GitHub](https://github.com/pavanamthomas), thomaspavanam@gmail.com).

The laboratory checks derivatives, not pretrained-model calling. Agreement on one MLP does not audit every PyTorch operator.

## Numerical checks

1. [`AUTOGRAD_VS_FINITE_DIFFERENCES.md`](AUTOGRAD_VS_FINITE_DIFFERENCES.md) — chain rule, central differences (truncation vs cancellation), autograd, ReLU at 0, ill-scaled softmax.
2. [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md) — failures the tests are required to keep visible.
3. [`src/dllab/numpy_net/`](src/dllab/numpy_net/) — MLP with no autograd. [`src/dllab/torch_net/compare.py`](src/dllab/torch_net/compare.py) — same init, same batch, NumPy vs PyTorch.
4. [`tests/`](tests/) — closed-form affine+ReLU+MSE, ε sweep, CE decrease on a linearly separable toy, BatchNorm train≠eval, clip bound, disjoint train/test indices, `state_dict` round-trip.
5. [`src/dllab/torch_net/cnn.py`](src/dllab/torch_net/cnn.py) and [`src/dllab/numpy_net/layers.py`](src/dllab/numpy_net/layers.py) — 1-D conv as a Toeplitz map (weight sharing, local receptive field) on a planted-motif DGP.
6. [`ROADMAP.md`](ROADMAP.md) — bounds and open checks.

Reproduce:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pytest
python scripts/run_all.py
```

Python 3.11 or newer. CPU torch is enough. CI is CPU.

## Implementation notes

- NumPy weights are `(in, out)` so `x @ W` is the map on the page. PyTorch `nn.Linear` is `(out, in)`. Comparison code transposes; it does not hope that two RNGs coincide.
- Softmax and cross-entropy are fused. The Jacobian is `(p - y)/N`. A separate softmax layer is an avoidable source of overflow.
- Gradient checking uses central differences and an ε sweep. A single ε is not a proof. ReLU at 0 is excluded from the "smooth map" claim.
- Adam with `weight_decay` is coupled L2. AdamW is decoupled. I compare them on a two-parameter quadratic so the difference is a path, not a slogan.
- The CNN exists to exhibit tied diagonals and a local window on a motif DGP. It is not an image classifier.
- Seeds go through `get_rng` / `seed_everything`. GPU nondeterminism is documented rather than "fixed" with a flag CI never exercises.

## Designed kinks and pathologies

- Zero initialisation: hidden units remain identical.
- Large initialisation: sigmoid local derivatives collapse; deep tanh gradients explode.
- Deep sigmoid stack: first-layer gradient norm shrinks with depth.
- Dead ReLU: a large negative bias zeros the incoming-weight gradient.
- BatchNorm: train-mode output on a size-4 batch is not eval-mode output.
- Unweighted CE on a 5%/95% Gaussian DGP: accuracy can look fine while minority recall does not.
- Finite differences: ε too large (truncation) and too small (cancellation); ReLU kink (not a \(C^2\) point).

Details: [`docs/failure_modes.md`](docs/failure_modes.md), [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md).

## Finite-difference and autograd agreement

Tests check mathematical properties, not merely that a function returns a tensor:

- Affine+ReLU+MSE reverse mode matches a closed form written in the test file.
- Central differences at ε=1e-5 agree with that analytic gradient on a tanh map; the ε sweep is worse at both ends.
- NumPy and PyTorch forwards and parameter gradients agree in float64 on a shared initialisation.
- Softmax CE decreases on a linearly separable toy DGP.
- `state_dict` save/load preserves parameters and a forward pass.
- After a short BatchNorm training loop, train vs eval disagree on a small probe batch.
- Global L2 clipping of `(3, 4)` yields `(0.6, 0.8)` and respects `max_norm`.
- `split_indices` returns a partition of `{0,...,n-1}` with empty intersection.

## Reproducibility

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pytest
python scripts/run_all.py
```

`scripts/run_all.py` writes figures under `outputs/figures/` and a numeric
summary to `outputs/tables/run_summary.csv`. Those files are regenerable.
The source of truth is the code plus the tests.

CI installs CPU PyTorch, installs the package, runs `pytest`, and runs
`scripts/run_all.py` with `MPLBACKEND=Agg`.

## Known limitations

- Every DGP is stylised. Two-moons accuracy is not a result about a natural dataset.
- Tiny epoch counts are for CI time, not for claiming that a model "trained."
- Matching autograd on one MLP does not audit every PyTorch operator.
- Weighted CE changes the finite-sample objective. It is not a sampling-design correction.
- No result here is a causal finding, a production latency number, or a pretrained-model benchmark.
- GPU bitwise reproducibility is not claimed. CI is CPU. See [`docs/reproducibility.md`](docs/reproducibility.md).

Related laboratories: [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation), [econometrics-causal-inference-lab](https://github.com/pavanamthomas/econometrics-causal-inference-lab).

## Remaining derivative-check bounds

Complex-step derivatives are not implemented; the default check is real
central differences. Residual networks, LayerNorm, and learning-rate
schedules are out of the current MLP stack. GPU bitwise reproducibility is
not claimed. See `ROADMAP.md`.

## Repository structure

```text
pytorch-deep-learning-lab/
├── AUTOGRAD_VS_FINITE_DIFFERENCES.md
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── docs/
├── src/dllab/
│   ├── numpy_net/
│   ├── gradcheck/
│   ├── torch_net/
│   ├── experiments/
│   └── data/
├── scripts/run_all.py
├── tests/
├── outputs/
└── .github/workflows/ci.yml
```

## Citation

See [`CITATION.cff`](CITATION.cff). Licence: MIT, Copyright 2026 Dr. Pavanam Thomas.
