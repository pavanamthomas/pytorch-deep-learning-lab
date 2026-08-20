# Failures and corrections

The laboratory keeps failure modes visible. A "successful" test here
often means the **wrong procedure still misbehaves** under a known DGP
or a known numerical hypothesis.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Central differences of ReLU at 0 | Quotient equals 1/2 for any ε>0; autograd uses 0 | `relu_kink_central_difference` | Do not treat a kink as a \(C^2\) check; probe away from 0 | `tests/test_finite_diff.py::test_relu_kink_central_difference_is_half` | Other kinks (abs, max-pool ties) |
| Finite differences with ε=0.1 or ε=1e-10 on tanh+MSE | Relative error much larger than at ε=1e-5 | Epsilon sweep U-shape | Choose ε from the sweep; do not trust a single ε | `tests/test_finite_diff.py::test_epsilon_sweep_u_shape` | Complex-step derivatives not implemented |
| Zero init of a hidden layer | All units compute the same (zero) map | `run_init_study` | Draw breaking symmetry (He/Xavier) | `docs/initialization.md`; init study in `scripts/run_all.py` | Whether a particular draw trains on a new DGP |
| Large Gaussian init into sigmoid | Mean local derivative collapses | `large_sigmoid_grad_mean` | Scale weights; avoid saturating starts | `run_init_study` | Depth at which even He fails |
| Dead ReLU via large negative bias | \(\|dW\|\approx 0\) while targets are nonzero | `run_activation_study` | Change bias/init or the activation | `run_activation_study` | Leakage ReLU variants |
| Deep sigmoid MLP | \(\|g_{\mathrm{first}}\|/\|g_{\mathrm{last}}\|\) shrinks with depth | `run_vanishing_study` | Different activation, residual maps, or less depth | `run_vanishing_study` | Residual nets not in this package |
| Large-init deep tanh | Global grad norm explodes | `run_exploding_and_clip` | Clip, or do not start there | `tests/test_clip.py` | Which clip threshold is "right" |
| BatchNorm eval on a size-4 probe after short training | Train output ≠ eval output | `run_batchnorm_study` | Call `eval()` at evaluation; do not treat the gap as a bug | `tests/test_batchnorm.py` | Running-stat bias at other momenta |
| Unweighted CE on 5%/95% Gaussians | Minority recall can stay poor while accuracy is high | `run_imbalance_study` | Inverse-prevalence weights, or a different utility | `run_imbalance_study` | Cost-sensitive thresholds in an application |
| Naive softmax without max-shift | Non-finite probabilities at huge logits | `ill_scaled_softmax_demo` | Stable log-softmax / fused CE | `dllab.gradcheck.three_way` | Temperature scaling as calibration |
| Shared `lr=0.05` for SGD and Adam on the 2-D quadratic | Adam's function value stayed above SGD's after 40 steps | Bias-corrected first moment; small shared step | Give Adam its own step size (`lr_adam=0.2`); do not rank methods at one lr | `tests/test_optimizers.py::test_shared_lr_is_not_a_method_ranking` | Schedules; other quadratics |
| Index split that could overlap | Would leak labels into training | Disjointness of `split_indices` | Use the helper; do not reshuffle across the cut | `tests/test_no_leakage.py` | Grouped or temporal splits |

Process: `docs/lab_process.md`. Algebra of the flagship check:
`AUTOGRAD_VS_FINITE_DIFFERENCES.md`.
