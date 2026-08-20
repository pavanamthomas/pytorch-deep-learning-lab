# Reproducibility

`dllab.experiments.reproducibility.seed_everything` seeds

- Python's `random`,
- NumPy's *global* RNG (`np.random.seed`),
- `torch.manual_seed`,
- `torch.cuda.manual_seed_all` if CUDA is present.

Synthetic DGPs in this package mostly use `numpy.random.Generator` via
`dllab._rng.get_rng`, which is independent of `np.random.seed` unless
the caller passes the integer seed into `get_rng`. Tests that need a
repeatable DGP pass `seed=2026` into the DGP function.

Default seed is `2026`.

## What is checked

Two CPU forwards of a tiny `TorchMLP` after `seed_everything(2026)`
match bit for bit (`run_reproducibility_check`). A second seed does not.

## What is not claimed

GPU training is not claimed to be deterministic. CUDA convolutions,
atomic adds in reductions, and cuDNN autotune can disagree across runs
with the same seed. `torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8` reduces the allowed kernel set; some
ops then error instead of running. This repository's CI is CPU. I do
not set those flags globally, because a green CI bar would then be
misread as a GPU warranty.

DataLoader workers (`num_workers>0`) have their own RNGs. The scripts
here use the default `num_workers=0`.

Figures and `outputs/tables/run_summary.csv` are regenerable. They are
not the source of truth. The source of truth is the code plus the tests.
