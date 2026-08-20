# Data policy

This repository is a neural-network laboratory. It does not ship
observational microdata, pretrained weights, or proprietary files.

## What is used

All arrays are **simulated** in `dllab.data.synthetic`: two-moons,
Gaussian mixtures, a linearly separable blob DGP, a 5%/95% imbalanced
Gaussian, and 1-D sequences with a planted motif. Randomness is
controlled through `numpy.random.Generator` with documented seeds
(default `2026`).

No file in `data/` is required. No download script is required.
HuggingFace is not a dependency.

## What is not claimed

In-sample accuracy, recall, and training-loss decrease describe the
behaviour of a procedure under a known DGP. They are not estimates for
a real population, a published empirical study, or a deployed model.

## Regeneration

Figures and tables under `outputs/` are disposable. They are written by
`python scripts/run_all.py` and are ignored by git except for `.gitkeep`
placeholders. A clean clone plus the commands in the README regenerates
them.

## Third-party code

The package depends on NumPy, pandas, matplotlib, pytest, and PyTorch
under their respective licences. This repository does not copy textbook
prose, exam questions, or copyrighted worked examples into `docs/`.
