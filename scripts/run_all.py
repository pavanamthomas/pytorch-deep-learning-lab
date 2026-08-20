"""Run the laboratory demonstrations and write figures and a summary table.

All numerical results printed here are simulations or closed-form arithmetic
on synthetic DGPs. They are not empirical findings about a real population.

Usage, from the repository root::

    python scripts/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dllab.data.synthetic import MOTIF, motif_sequences, two_moons
from dllab.experiments.activations import run_activation_study
from dllab.experiments.batchnorm import run_batchnorm_study
from dllab.experiments.gradients import run_exploding_and_clip, run_vanishing_study
from dllab.experiments.imbalance import run_imbalance_study
from dllab.experiments.initialization import run_init_study
from dllab.experiments.optimizers import run_adam_vs_adamw, run_quadratic_paths
from dllab.experiments.regularization import run_dropout_study, run_early_stopping_study, run_l2_study
from dllab.experiments.reproducibility import run_reproducibility_check
from dllab.gradcheck.finite_diff import relu_kink_central_difference
from dllab.gradcheck.three_way import ill_scaled_softmax_demo, run_three_way_check, tanh_mse_epsilon_sweep
from dllab.numpy_net.layers import conv1d_forward, conv1d_toeplitz
from dllab.numpy_net.mlp import NumpyMLP, train_softmax_ce
from dllab.plots import (
    plot_epsilon_sweep,
    plot_imbalance_recall,
    plot_quadratic_paths,
    plot_two_moons_points,
    plot_vanishing,
)
from dllab.torch_net.cnn import MotifCNN
from dllab.torch_net.compare import compare_mse_batch, compare_softmax_ce_batch
from dllab.torch_net.dataset import ArrayPairDataset


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _train_motif_cnn(seed: int = 2026, epochs: int = 12) -> dict[str, float]:
    data = motif_sequences(n=48, length=16, seed=seed)
    ds = ArrayPairDataset(data.x, data.y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    torch.manual_seed(seed)
    model = MotifCNN(n_filters=4, kernel_size=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.08)
    loss_fn = nn.CrossEntropyLoss()
    last = 0.0
    first = None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if first is None:
                first = float(loss.item())
            loss.backward()
            opt.step()
            last = float(loss.item())
    assert first is not None
    # Kernel–motif cosine on the first filter, as a descriptive number only.
    k = model.conv.weight.detach().cpu().numpy()[0, 0]
    motif = MOTIF
    cos = float(np.dot(k, motif) / (np.linalg.norm(k) * np.linalg.norm(motif) + 1e-12))
    return {"loss_first": first, "loss_last": last, "kernel_motif_cosine": cos}


def main() -> None:
    fig_dir = ROOT / "outputs" / "figures"
    tab_dir = ROOT / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    _print_header("A. Manual derivative vs finite differences vs autograd")
    three = run_three_way_check(seed=2026)
    print(
        f"affine+tanh+MSE: L={three.loss:.6f}, eps={three.chosen_epsilon:g}, "
        f"rel(fd, analytic)={three.rel_fd_vs_analytic:.2e}, "
        f"rel(autograd, analytic)={three.rel_autograd_vs_analytic:.2e}"
    )
    print("Algebra is in AUTOGRAD_VS_FINITE_DIFFERENCES.md. These numbers are this run.")
    rows.append({"quantity": "three_way_rel_fd_analytic", "value": three.rel_fd_vs_analytic})
    rows.append({"quantity": "three_way_rel_auto_analytic", "value": three.rel_autograd_vs_analytic})

    sweep = tanh_mse_epsilon_sweep(seed=2026)
    eps = np.array([p.epsilon for p in sweep], dtype=float)
    rel = np.array([p.relative_error for p in sweep], dtype=float)
    plot_epsilon_sweep(eps, rel, fig_dir / "epsilon_sweep.png")
    best = sweep[int(np.argmin(rel))]
    worst_large = sweep[0]
    worst_small = sweep[-1]
    print(
        f"epsilon sweep: best rel err {best.relative_error:.2e} at eps={best.epsilon:g}; "
        f"eps={worst_large.epsilon:g} -> {worst_large.relative_error:.2e}; "
        f"eps={worst_small.epsilon:g} -> {worst_small.relative_error:.2e}"
    )
    print("Large eps: Taylor truncation. Small eps: subtractive cancellation.")
    rows.append({"quantity": "fd_rel_best", "value": best.relative_error})
    rows.append({"quantity": "fd_rel_eps_1e-1", "value": worst_large.relative_error})
    rows.append({"quantity": "fd_rel_eps_1e-10", "value": worst_small.relative_error})

    kink = relu_kink_central_difference(epsilon=1e-5)
    print(
        f"ReLU at 0: central diff={kink['central_difference']:.3f} "
        f"(autograd convention {kink['autograd_convention']}). "
        "The gap does not vanish with eps."
    )
    soft = ill_scaled_softmax_demo()
    print(
        f"naive softmax at huge logits finite={soft['naive_bad_finite']}; "
        f"stable softmax peak={soft['stable_bad_peak']:.6f}; "
        f"CE at saturated logits={soft['ce_at_saturated_logits']:.2e}"
    )
    rows.append({"quantity": "relu_kink_central_diff", "value": kink["central_difference"]})

    _print_header("B. NumPy reverse mode vs PyTorch autograd (shared init, one batch)")
    mse_ag = compare_mse_batch(seed=2026)
    ce_ag = compare_softmax_ce_batch(seed=2026)
    print(
        f"MSE tanh MLP: max|forward|={mse_ag.max_abs_forward_diff:.2e}, "
        f"max|dW|={mse_ag.max_abs_weight_grad_diff:.2e}"
    )
    print(
        f"softmax CE tanh MLP: max|forward|={ce_ag.max_abs_forward_diff:.2e}, "
        f"max|dW|={ce_ag.max_abs_weight_grad_diff:.2e}"
    )
    rows.append({"quantity": "mse_forward_diff", "value": mse_ag.max_abs_forward_diff})
    rows.append({"quantity": "mse_grad_diff", "value": mse_ag.max_abs_weight_grad_diff})
    rows.append({"quantity": "ce_forward_diff", "value": ce_ag.max_abs_forward_diff})
    rows.append({"quantity": "ce_grad_diff", "value": ce_ag.max_abs_weight_grad_diff})

    _print_header("C. Initialisation and activations")
    init = run_init_study(seed=2026)
    print(
        f"zero init: hidden units identical={init.zero_units_identical}, "
        f"std across units={init.zero_hidden_std_across_units:.2e}"
    )
    print(
        f"large Gaussian (scale=8): mean |pre-act|={init.large_preact_abs_mean:.3f}, "
        f"mean sigmoid'={init.large_sigmoid_grad_mean:.2e}"
    )
    print(
        f"Xavier tanh hidden var={init.xavier_tanh_hidden_var:.3f}; "
        f"He ReLU hidden var={init.he_relu_hidden_var:.3f}"
    )
    act = run_activation_study(seed=2026)
    print(
        f"sigmoid' mean at N(0,1)={act.sigmoid_grad_at_scale_1:.3f}, "
        f"at N(0,8^2)={act.sigmoid_grad_at_scale_8:.3e}"
    )
    print(
        f"dead ReLU: fraction of zeros={act.dead_relu_fraction_zero:.3f}, "
        f"max |dW|={act.dead_relu_weight_grad_abs_max:.2e}"
    )
    rows.append({"quantity": "dead_relu_dW_max", "value": act.dead_relu_weight_grad_abs_max})
    rows.append({"quantity": "large_init_sigmoid_grad", "value": init.large_sigmoid_grad_mean})

    _print_header("D. Optimisers on an ill-conditioned quadratic")
    paths = run_quadratic_paths(steps=40, lr_sgd=0.05, lr_adam=0.2)
    plot_quadratic_paths(paths.sgd, paths.momentum, paths.adam, paths.adamw, fig_dir / "quadratic_paths.png")
    decay = run_adam_vs_adamw(steps=80)
    print(
        f"quadratic f after 40 SGD steps (lr=0.05)={paths.values_sgd[-1]:.4f}, "
        f"after 40 Adam steps (lr=0.2)={paths.values_adam[-1]:.4f}"
    )
    print("Those two numbers are not a ranking: the step sizes differ on purpose.")
    print(
        f"Adam vs AdamW with lambda=0.2: ||w||_Adam={decay.adam_param_norm:.4f}, "
        f"||w||_AdamW={decay.adamw_param_norm:.4f}; "
        f"coord-decay ratio Adam={decay.coordinate_decay_ratio_adam:.3f}, "
        f"AdamW={decay.coordinate_decay_ratio_adamw:.3f}"
    )
    print("Coupled L2 is scaled by 1/sqrt(v). Decoupled decay is not.")
    rows.append({"quantity": "adam_wd_param_norm", "value": decay.adam_param_norm})
    rows.append({"quantity": "adamw_param_norm", "value": decay.adamw_param_norm})

    _print_header("E. Regularisation, BatchNorm, gradients")
    l2 = run_l2_study(seed=2026)
    drop = run_dropout_study(seed=2026)
    early = run_early_stopping_study(seed=2026)
    print(f"L2: ||theta|| unregularised={l2.unregularised_weight_norm:.3f}, with L2={l2.l2_weight_norm:.3f}")
    print(
        f"inverted dropout p=0.5: train fraction exact zero={drop.train_frac_exact_zero:.3f} "
        f"(eval has no zeros from dropout)"
    )
    print(
        f"early stopping: best val epoch={early.best_epoch}, "
        f"val_best={early.val_loss_at_best:.4f}, val_final={early.val_loss_final:.4f}, "
        f"val rose after best={early.val_rose_after_best}"
    )
    bn = run_batchnorm_study(seed=2026)
    print(
        f"BatchNorm1d: batch_size={bn.batch_size}, "
        f"max |train-eval| on a probe batch={bn.max_abs_train_eval_diff:.4f}; "
        f"probe mean={bn.batch_mean:.3f}, running mean={bn.running_mean:.3f}"
    )
    vanish_rows = []
    for d in (3, 5, 7, 9):
        v = run_vanishing_study(depth=d, seed=2026)
        vanish_rows.append((d, v.ratio_first_to_last))
        print(
            f"sigmoid MLP depth={d}: ||g_first||/||g_last||={v.ratio_first_to_last:.3e} "
            f"(||g_first||={v.first_layer_grad_norm:.3e})"
        )
    plot_vanishing(
        np.array([r[0] for r in vanish_rows], dtype=float),
        np.array([r[1] for r in vanish_rows], dtype=float),
        fig_dir / "vanishing_gradients.png",
    )
    exp = run_exploding_and_clip(seed=2026)
    print(
        f"large-init tanh stack: unclipped ||g||={exp.unclipped_norm:.3e}, "
        f"clipped to {exp.max_norm} -> {exp.clipped_norm:.3e}, bound ok={exp.respected_bound}"
    )
    rows.append({"quantity": "bn_train_eval_diff", "value": bn.max_abs_train_eval_diff})
    rows.append({"quantity": "clip_unclipped_norm", "value": exp.unclipped_norm})
    rows.append({"quantity": "clip_clipped_norm", "value": exp.clipped_norm})

    _print_header("F. Class imbalance (5%/95% Gaussian DGP)")
    imb = run_imbalance_study(seed=2026)
    print(
        f"prevalence={imb.prevalence:.3f}: unweighted recall={imb.unweighted_recall:.3f} "
        f"(acc={imb.unweighted_accuracy:.3f}); weighted recall={imb.weighted_recall:.3f} "
        f"(acc={imb.weighted_accuracy:.3f})"
    )
    print("Recall and accuracy are in-sample on the same DGP draw. Not a deployment metric.")
    plot_imbalance_recall(imb.unweighted_recall, imb.weighted_recall, fig_dir / "imbalance_recall.png")
    rows.append({"quantity": "imbalance_unweighted_recall", "value": imb.unweighted_recall})
    rows.append({"quantity": "imbalance_weighted_recall", "value": imb.weighted_recall})

    _print_header("G. Models: two-moons MLP and motif CNN")
    moons = two_moons(n=80, seed=2026)
    plot_two_moons_points(moons.x, moons.y, fig_dir / "two_moons.png")
    mlp = NumpyMLP.from_sizes([2, 8, 2], activation="relu", init="he", seed=2026)
    hist = train_softmax_ce(mlp, moons.x, moons.y, lr=0.25, epochs=25)
    pred = np.argmax(mlp.forward(moons.x), axis=1)
    acc = float(np.mean(pred == moons.y))
    print(f"two-moons MLP: CE {hist[0]:.3f} -> {hist[-1]:.3f}; in-sample acc={acc:.3f}")
    print("In-sample accuracy on a simulated DGP is not a generalisation claim.")

    seq = np.zeros((1, 1, 8), dtype=float)
    seq[0, 0, 2:5] = MOTIF
    w = MOTIF.reshape(1, 1, 3)
    conv_out = conv1d_forward(seq, w)
    toe = conv1d_toeplitz(w, length=8)
    equiv = toe @ seq.reshape(-1)
    print(
        f"conv1d vs Toeplitz: max |diff|={np.max(np.abs(conv_out.reshape(-1) - equiv)):.2e}; "
        "tied diagonals are weight sharing."
    )
    cnn = _train_motif_cnn(seed=2026, epochs=12)
    print(
        f"motif CNN: batch CE {cnn['loss_first']:.3f} -> {cnn['loss_last']:.3f}; "
        f"filter-0 cosine with planted motif={cnn['kernel_motif_cosine']:.3f} "
        "(descriptive; not a recovery guarantee)"
    )
    rows.append({"quantity": "two_moons_ce_final", "value": hist[-1]})
    rows.append({"quantity": "two_moons_in_sample_acc", "value": acc})
    rows.append({"quantity": "cnn_ce_final", "value": cnn["loss_last"]})

    _print_header("H. Reproducibility")
    rep = run_reproducibility_check(seed=2026)
    print(f"CPU forwards match after seed_everything(2026): {rep.cpu_forwards_match}")
    print(f"CUDA available: {rep.cuda_available}")
    print(rep.warning)
    rows.append({"quantity": "cpu_forwards_match", "value": float(rep.cpu_forwards_match)})

    summary = pd.DataFrame(rows)
    out_csv = tab_dir / "run_summary.csv"
    summary.to_csv(out_csv, index=False)
    print()
    print(f"Wrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote figures under {fig_dir.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
