"""Designed experiments: init, activations, optimisers, regularisation, BN, grads, imbalance."""

from dllab.experiments.activations import ActivationStudy, run_activation_study
from dllab.experiments.batchnorm import BatchNormStudy, run_batchnorm_study
from dllab.experiments.gradients import ExplodingStudy, VanishingStudy, run_exploding_and_clip, run_vanishing_study
from dllab.experiments.imbalance import ImbalanceStudy, run_imbalance_study
from dllab.experiments.initialization import InitStudy, run_init_study
from dllab.experiments.optimizers import AdamVsAdamW, QuadraticPaths, run_adam_vs_adamw, run_quadratic_paths
from dllab.experiments.regularization import (
    DropoutStudy,
    EarlyStoppingStudy,
    L2Study,
    run_dropout_study,
    run_early_stopping_study,
    run_l2_study,
)
from dllab.experiments.reproducibility import ReproducibilityNote, seed_everything

__all__ = [
    "ActivationStudy",
    "AdamVsAdamW",
    "BatchNormStudy",
    "DropoutStudy",
    "EarlyStoppingStudy",
    "ExplodingStudy",
    "ImbalanceStudy",
    "InitStudy",
    "L2Study",
    "QuadraticPaths",
    "ReproducibilityNote",
    "VanishingStudy",
    "run_activation_study",
    "run_adam_vs_adamw",
    "run_batchnorm_study",
    "run_dropout_study",
    "run_early_stopping_study",
    "run_exploding_and_clip",
    "run_imbalance_study",
    "run_init_study",
    "run_l2_study",
    "run_quadratic_paths",
    "run_vanishing_study",
    "seed_everything",
]
