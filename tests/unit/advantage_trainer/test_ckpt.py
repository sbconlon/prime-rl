"""AdvTrainer checkpoint save/load round-trip tests.

Crash recovery is the load-bearing property here: if save silently
fails or load doesn't preserve the trainable state exactly, a crash
mid-run loses all learning progress. These tests are the smoke check
that the basic path actually works.

Four tests:
  1. save() writes the expected files to disk (value_state.pt + STABLE)
  2. save() + load() round-trip preserves trainable param values exactly
  3. save() + load() round-trip preserves optimizer state (Adam moments)
  4. maybe_clean() with keep_last=1 deletes older checkpoints on disk
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.ckpt import (
    AdvantageCheckpointManager,
    setup_advantage_ckpt_manager,
)
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_backbone(seed: int = 0) -> ValueNetworkBackbone:
    """Same fixture other AdvTrainer tests use, with non-zero LoRA + heads."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    backbone.train()
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.copy_(torch.randn_like(p) * 0.1)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def _trainable_snapshot(backbone: ValueNetworkBackbone) -> dict[str, torch.Tensor]:
    """Detached CPU clone of every trainable param, keyed by name."""
    return {
        name: p.detach().cpu().clone()
        for name, p in backbone.named_parameters()
        if p.requires_grad
    }


# ---------------------------------------------------------------------------
# 1. save() writes the expected files to disk
# ---------------------------------------------------------------------------


def test_save_writes_value_state_and_stable(tmp_path: Path):
    """After save(step=5, ...), the expected files exist on disk."""
    ckpt_config = CheckpointConfig(interval=1, resume_step=None)
    mgr = AdvantageCheckpointManager(tmp_path, ckpt_config)

    backbone = _tiny_backbone()
    optimizer = optim.AdamW([p for p in backbone.parameters() if p.requires_grad], lr=1e-3)

    mgr.save(step=5, backbone=backbone, optimizer=optimizer)

    expected_value_state = tmp_path / "checkpoints" / "step_5" / "trainer" / "value_state.pt"
    expected_stable = tmp_path / "checkpoints" / "step_5" / "STABLE"

    assert expected_value_state.exists(), f"value_state.pt not written to {expected_value_state}"
    assert expected_value_state.stat().st_size > 0, "value_state.pt is empty"
    assert expected_stable.exists(), f"STABLE sentinel not written to {expected_stable}"

    # The manager's ckpt_steps list is updated post-save.
    assert 5 in mgr.ckpt_steps


# ---------------------------------------------------------------------------
# 2. save() + load() preserves trainable params exactly
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_trainable_params(tmp_path: Path):
    """Save backbone A, load into backbone B (different init), verify all
    trainable params match A's exactly. This is the load-bearing crash-
    recovery property."""
    ckpt_config = CheckpointConfig(interval=1, resume_step=None)
    mgr = AdvantageCheckpointManager(tmp_path, ckpt_config)

    # Backbone A: seed=0, has known LoRA + head values
    backbone_a = _tiny_backbone(seed=0)
    optimizer_a = optim.AdamW(
        [p for p in backbone_a.parameters() if p.requires_grad], lr=1e-3
    )
    snapshot_before = _trainable_snapshot(backbone_a)
    mgr.save(step=7, backbone=backbone_a, optimizer=optimizer_a)

    # Backbone B: seed=42, totally different LoRA + head values.
    backbone_b = _tiny_backbone(seed=42)
    optimizer_b = optim.AdamW(
        [p for p in backbone_b.parameters() if p.requires_grad], lr=1e-3
    )

    # Confirm A and B genuinely differ before load (sanity).
    snapshot_b_before = _trainable_snapshot(backbone_b)
    differs_count = sum(
        1 for name in snapshot_before
        if not torch.equal(snapshot_before[name], snapshot_b_before[name])
    )
    assert differs_count > 0, "Test setup: backbones A and B should differ before load"

    # Load A's checkpoint into B.
    loaded_step = mgr.load(step=7, backbone=backbone_b, optimizer=optimizer_b)
    assert loaded_step == 7, f"load() returned wrong step: {loaded_step}"

    # All trainable params in B should now match A.
    snapshot_after = _trainable_snapshot(backbone_b)
    for name, tensor_a in snapshot_before.items():
        tensor_b = snapshot_after[name]
        assert torch.equal(tensor_a, tensor_b), (
            f"Trainable param {name} differs after load: "
            f"A_norm={tensor_a.norm().item():.4e}, B_norm={tensor_b.norm().item():.4e}, "
            f"diff_norm={(tensor_a - tensor_b).norm().item():.4e}"
        )


# ---------------------------------------------------------------------------
# 3. Optimizer state is preserved
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_optimizer_state(tmp_path: Path):
    """Save optimizer with non-trivial Adam moments, load into fresh
    optimizer, verify moments match."""
    ckpt_config = CheckpointConfig(interval=1, resume_step=None, skip_optimizer=False)
    mgr = AdvantageCheckpointManager(tmp_path, ckpt_config)

    backbone_a = _tiny_backbone(seed=0)
    trainable_a = [p for p in backbone_a.parameters() if p.requires_grad]
    optimizer_a = optim.AdamW(trainable_a, lr=1e-3)

    # Drive a few steps so Adam moments are non-trivial.
    for _ in range(3):
        loss = sum(p.pow(2).sum() for p in trainable_a)
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

    # Capture A's state-dict for direct comparison.
    state_a = optimizer_a.state_dict()

    mgr.save(step=3, backbone=backbone_a, optimizer=optimizer_a)

    # Fresh backbone + optimizer (no training steps -> empty moments).
    backbone_b = _tiny_backbone(seed=99)
    trainable_b = [p for p in backbone_b.parameters() if p.requires_grad]
    optimizer_b = optim.AdamW(trainable_b, lr=1e-3)
    state_b_before = optimizer_b.state_dict()
    # Sanity: optimizer B has no Adam moments yet.
    assert len(state_b_before["state"]) == 0, "Fresh optimizer should have empty state"

    mgr.load(step=3, backbone=backbone_b, optimizer=optimizer_b)

    state_b_after = optimizer_b.state_dict()
    # The state dict's "state" key holds per-parameter momentum tensors.
    assert len(state_b_after["state"]) == len(state_a["state"]), (
        f"Optimizer state count mismatch: A={len(state_a['state'])} "
        f"B={len(state_b_after['state'])}"
    )

    # Compare per-parameter Adam state tensors (exp_avg, exp_avg_sq, step).
    for pid in state_a["state"]:
        a_state = state_a["state"][pid]
        b_state = state_b_after["state"][pid]
        for key in ["exp_avg", "exp_avg_sq"]:
            if key not in a_state:
                continue
            assert torch.equal(a_state[key], b_state[key]), (
                f"Optimizer state '{key}' for param {pid} differs after load"
            )


# ---------------------------------------------------------------------------
# 4. maybe_clean() retention policy
# ---------------------------------------------------------------------------


def test_maybe_clean_retains_keep_last(tmp_path: Path):
    """Save 5 checkpoints with keep_last=2; verify only the 2 most recent
    remain on disk after maybe_clean()."""
    ckpt_config = CheckpointConfig(
        interval=1, resume_step=None, keep_last=2, keep_interval=None
    )
    mgr = AdvantageCheckpointManager(tmp_path, ckpt_config)

    backbone = _tiny_backbone()
    optimizer = optim.AdamW(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-3
    )

    for step in [10, 20, 30, 40, 50]:
        mgr.save(step=step, backbone=backbone, optimizer=optimizer)
        mgr.maybe_clean()

    remaining_steps = sorted(mgr.ckpt_steps)
    assert remaining_steps == [40, 50], (
        f"Expected only [40, 50] after keep_last=2, got {remaining_steps}"
    )

    # Verify physically on disk too.
    ckpt_root = tmp_path / "checkpoints"
    on_disk_steps = sorted(
        int(d.name.replace("step_", ""))
        for d in ckpt_root.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    )
    assert on_disk_steps == [40, 50], (
        f"Disk state inconsistent: expected [40, 50], found {on_disk_steps}"
    )


# ---------------------------------------------------------------------------
# 5. setup_advantage_ckpt_manager returns None for None config
# ---------------------------------------------------------------------------


def test_setup_returns_none_when_config_is_none(tmp_path: Path):
    """setup_advantage_ckpt_manager(_, None) returns None so downstream
    `if ckpt_manager is not None` guards no-op cleanly. Matches LLM
    trainer's setup_ckpt_managers pattern."""
    result = setup_advantage_ckpt_manager(tmp_path, None)
    assert result is None
