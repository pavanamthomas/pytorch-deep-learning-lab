"""Save and load of a TorchMLP state_dict preserves parameters and forward values."""

from __future__ import annotations

from pathlib import Path

import torch

from dllab.torch_net.mlp import TorchMLP


def test_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    model = TorchMLP([3, 5, 2], activation="tanh")
    x = torch.randn(4, 3)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()
    path = tmp_path / "mlp.pt"
    torch.save(model.state_dict(), path)

    restored = TorchMLP([3, 5, 2], activation="tanh")
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        after = restored(x)
    assert torch.allclose(before, after)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
