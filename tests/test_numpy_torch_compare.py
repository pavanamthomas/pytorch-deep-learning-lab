"""NumPy MLP and Torch MLP agree on a controlled batch."""

from __future__ import annotations

from dllab.torch_net.compare import compare_mse_batch, compare_softmax_ce_batch


def test_numpy_torch_mse_agreement() -> None:
    result = compare_mse_batch(seed=2026)
    assert result.max_abs_forward_diff < 1e-10
    assert result.max_abs_loss_diff < 1e-10
    assert result.max_abs_weight_grad_diff < 1e-10


def test_numpy_torch_softmax_ce_agreement() -> None:
    result = compare_softmax_ce_batch(seed=2026)
    assert result.max_abs_forward_diff < 1e-10
    assert result.max_abs_loss_diff < 1e-10
    assert result.max_abs_weight_grad_diff < 1e-10
