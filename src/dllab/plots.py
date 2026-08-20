"""Figures for the laboratory scripts. Labels are simulation annotations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_epsilon_sweep(epsilons: NDArray, rel_errors: NDArray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.loglog(epsilons, rel_errors, marker="o")
    ax.set_xlabel("finite-difference epsilon")
    ax.set_ylabel("relative error vs analytic gradient")
    ax.set_title("Central differences: truncation vs cancellation (synthetic tanh+MSE)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_quadratic_paths(
    sgd: NDArray,
    momentum: NDArray,
    adam: NDArray,
    adamw: NDArray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(sgd[:, 0], sgd[:, 1], label="SGD")
    ax.plot(momentum[:, 0], momentum[:, 1], label="SGD+momentum")
    ax.plot(adam[:, 0], adam[:, 1], label="Adam")
    ax.plot(adamw[:, 0], adamw[:, 1], label="AdamW (wd=0)")
    ax.scatter([0.0], [0.0], c="k", marker="x", label="minimiser")
    ax.set_xlabel("w0 (weak curvature)")
    ax.set_ylabel("w1 (strong curvature)")
    ax.set_title("Ill-conditioned quadratic (simulation, two parameters)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_vanishing(depths: NDArray, ratios: NDArray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogy(depths, ratios, marker="o")
    ax.set_xlabel("depth (number of affine maps)")
    ax.set_ylabel("||dL/dW_first|| / ||dL/dW_last||")
    ax.set_title("Deep sigmoid MLP, Xavier init (synthetic Gaussian batch)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_imbalance_recall(unweighted: float, weighted: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(["unweighted CE", "inverse-prevalence weights"], [unweighted, weighted], color=["#4c72b0", "#dd8452"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("minority-class recall (in-sample)")
    ax.set_title("5%/95% Gaussian DGP, linear head, short fit")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_two_moons_points(x: NDArray, y: NDArray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(x[y == 0, 0], x[y == 0, 1], s=18, label="class 0")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], s=18, label="class 1")
    ax.set_title("Two-moons DGP (simulated)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
