"""BatchNorm train-mode output is not eval-mode output on a small probe batch."""

from __future__ import annotations

from dllab.experiments.batchnorm import run_batchnorm_study


def test_batchnorm_train_eval_differ() -> None:
    study = run_batchnorm_study(seed=2026, batch_size=4, steps=15)
    assert study.max_abs_train_eval_diff > 1e-3
    assert study.batch_size == 4
