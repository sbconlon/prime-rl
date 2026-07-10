"""Phase 4: the warm-start load seam. load_warm_start unwraps the value_state.pt
["trainable_state_dict"] payload and loads it strict=False (frozen base absent)."""

import torch
from torch import nn

from prime_rl.advantage_trainer.ckpt import load_warm_start


class _TinyBackbone(nn.Module):
    """A trainable head + a frozen 'base' -- mimics the trainable/frozen split of
    ValueNetworkBackbone for load_warm_start's purposes."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)
        self.frozen = nn.Linear(2, 2)
        for p in self.frozen.parameters():
            p.requires_grad_(False)


def _save_value_state(module, path):
    trainable = {
        name: p.detach().cpu() for name, p in module.named_parameters() if p.requires_grad
    }
    torch.save({"step": 0, "trainable_state_dict": trainable}, path)


def test_load_warm_start_unwraps_and_loads(tmp_path):
    src = _TinyBackbone()
    with torch.no_grad():
        src.head.weight.fill_(0.5)
        src.head.bias.fill_(0.25)
    path = tmp_path / "value_state.pt"
    _save_value_state(src, path)

    dst = _TinyBackbone()
    with torch.no_grad():
        dst.head.weight.zero_()
        dst.head.bias.zero_()
    load_warm_start(dst, path)

    assert torch.allclose(dst.head.weight, torch.full_like(dst.head.weight, 0.5))
    assert torch.allclose(dst.head.bias, torch.full_like(dst.head.bias, 0.25))


def test_load_warm_start_ignores_frozen_base(tmp_path):
    src = _TinyBackbone()
    path = tmp_path / "value_state.pt"
    _save_value_state(src, path)

    dst = _TinyBackbone()
    frozen_before = dst.frozen.weight.detach().clone()
    load_warm_start(dst, path)  # strict=False: no crash despite absent frozen keys

    assert torch.allclose(dst.frozen.weight, frozen_before)
