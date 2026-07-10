"""Advantage Trainer checkpoint manager.

Mirrors the LLM Trainer's CheckpointManager (prime_rl.trainer.ckpt) in
public API + file layout + config-field semantics, but adapted for the
single-GPU AdvTrainer:

  - Saves only trainable parameters (LoRA slots + value heads) plus
    optimizer state. The frozen base is reconstructed at startup from
    the SFT checkpoint, so saving it would be wasted disk.
  - Uses plain torch.save (no torch.distributed.checkpoint -- the
    AdvTrainer doesn't initialize a distributed group).
  - No scheduler, no dataloader, no DTensor handling -- single GPU.

File layout matches the LLM trainer's:

    <output_dir>/
      checkpoints/
        step_<N>/
          trainer/
            value_state.pt
          STABLE                 # written last; signals "complete"

resume_step semantics match the LLM trainer's CheckpointConfig:

    None  -- start fresh
    -1    -- resume from latest available checkpoint
     N    -- resume from step N specifically
"""

from __future__ import annotations

import bisect
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path
from prime_rl.utils.utils import resolve_latest_ckpt_step


def _trainable_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    """Per-name CPU tensors for all params with requires_grad=True.

    Matches the format that AdvSrv's /update_weights endpoint expects
    (load_state_dict(strict=False) tolerates the missing frozen-base
    keys).
    """
    return {
        name: param.detach().cpu()
        for name, param in backbone.named_parameters()
        if param.requires_grad
    }


class AdvantageCheckpointManager:
    """Save / load AdvTrainer state at training-step granularity.

    Mirrors prime_rl.trainer.ckpt.CheckpointManager's public surface so
    train.py can use the same conditional patterns. Single-GPU version:
    uses torch.save / torch.load rather than dcp_save / dcp_load.
    """

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.skip_optimizer = config.skip_optimizer
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()

        all_steps = get_all_ckpt_steps(self.ckpt_dir)
        if config.resume_step is not None and config.resume_step >= 0:
            self.ckpt_steps = [s for s in all_steps if s <= config.resume_step]
        else:
            self.ckpt_steps = list(all_steps)

    # ---- path helpers (mirror CheckpointManager) ----

    def get_ckpt_path(self, step: int) -> Path:
        """Path to the AdvTrainer checkpoint subdir for a given step."""
        return get_step_path(self.ckpt_dir, step) / "trainer"

    def mark_stable(self, step: int) -> None:
        """Write STABLE sentinel so any concurrent reader knows the
        checkpoint is fully flushed."""
        step_path = get_step_path(self.ckpt_dir, step)
        (step_path / "STABLE").touch()

    # ---- save / load ----

    def save_to_path(
        self,
        path: Path,
        backbone: nn.Module,
        optimizer: Optimizer,
        step: int,
    ) -> None:
        """Write trainable params + optimizer state to a specific path."""
        self.logger.debug(f"Saving AdvTrainer checkpoint to {path}")
        start_time = time.perf_counter()
        path.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "step": step,
            "trainable_state_dict": _trainable_state_dict(backbone),
        }
        if not self.skip_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(payload, path / "value_state.pt")
        self.logger.debug(
            f"AdvTrainer checkpoint saved in {time.perf_counter() - start_time:.2f}s"
        )

    def load_from_path(
        self,
        path: Path,
        backbone: nn.Module,
        optimizer: Optimizer,
    ) -> int:
        """Load trainable params (+ optimizer state if not skipped) from a
        specific path. Returns the step number stored in the checkpoint."""
        self.logger.debug(f"Loading AdvTrainer checkpoint from {path}")
        start_time = time.perf_counter()

        payload = torch.load(path / "value_state.pt", map_location="cpu", weights_only=False)

        # Move trainable tensors to the backbone's device + dtype before loading.
        # Use the first trainable param as the device/dtype reference.
        first_trainable = next(p for p in backbone.parameters() if p.requires_grad)
        target_device = first_trainable.device
        target_dtype = first_trainable.dtype
        state_dict = {
            name: t.to(device=target_device, dtype=target_dtype)
            for name, t in payload["trainable_state_dict"].items()
        }

        # strict=False because the frozen-base keys aren't in our payload --
        # they're already loaded from the SFT checkpoint at startup.
        result = backbone.load_state_dict(state_dict, strict=False)
        unexpected_count = len(result.unexpected_keys)
        if unexpected_count > 0:
            self.logger.warning(
                f"AdvTrainer load: {unexpected_count} unexpected keys in checkpoint "
                f"(first few: {result.unexpected_keys[:3]}). Continuing."
            )

        if not self.skip_optimizer and "optimizer_state_dict" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(
                    f"AdvTrainer load: optimizer state restore failed ({e}); "
                    f"continuing with fresh optimizer state (Adam moments will re-warm)."
                )

        self.logger.debug(
            f"AdvTrainer checkpoint loaded in {time.perf_counter() - start_time:.2f}s"
        )
        return int(payload["step"])

    # ---- per-step convenience wrappers (mirror CheckpointManager) ----

    def save(self, step: int, backbone: nn.Module, optimizer: Optimizer) -> None:
        """Save the full AdvTrainer checkpoint for a specific step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_to_path(ckpt_path, backbone, optimizer, step)
        self.mark_stable(step)
        bisect.insort(self.ckpt_steps, step)

    def load(self, step: int, backbone: nn.Module, optimizer: Optimizer) -> int:
        """Load the AdvTrainer checkpoint at a specific step."""
        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"AdvTrainer checkpoint not found at {ckpt_path}")
        return self.load_from_path(ckpt_path, backbone, optimizer)

    # ---- retention policy (verbatim from CheckpointManager) ----

    def maybe_clean(self) -> None:
        """Apply keep_last + keep_interval retention. No-op if both are None."""
        if self.config.keep_last is None and self.config.keep_interval is None:
            return

        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)

        steps_to_keep: set[int] = set()
        if self.config.keep_last is not None:
            steps_to_keep.update(self.ckpt_steps[-self.config.keep_last:])
        if self.config.keep_interval is not None:
            for s in self.ckpt_steps:
                if s % self.config.keep_interval == 0:
                    steps_to_keep.add(s)

        to_delete = [s for s in self.ckpt_steps if s not in steps_to_keep]
        for s in to_delete:
            step_root = get_step_path(self.ckpt_dir, s)
            if step_root.exists():
                self.logger.debug(f"Removing past AdvTrainer checkpoint for step {s} ({step_root})")
                import shutil
                try:
                    shutil.rmtree(step_root)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {step_root}: {e}")

        self.ckpt_steps = [s for s in self.ckpt_steps if s in steps_to_keep]


def setup_advantage_ckpt_manager(
    output_dir: Path, ckpt_config: CheckpointConfig | None
) -> AdvantageCheckpointManager | None:
    """Mirror of setup_ckpt_managers for the AdvTrainer.

    Returns None when ckpt config is None (no checkpointing configured),
    matching the LLM trainer's pattern where downstream code conditionally
    skips ckpt save/load with `if ckpt_manager is not None`.
    """
    if ckpt_config is None:
        return None
    base_dir = ckpt_config.output_dir or output_dir
    return AdvantageCheckpointManager(base_dir, ckpt_config)


def load_warm_start(backbone: nn.Module, path: str | Path) -> None:
    """Load warm-start value weights (LoRA slots + value heads) from a Phase-3
    value_state.pt into `backbone`. Unwraps the ["trainable_state_dict"] payload
    (the AdvantageCheckpointManager format, NOT a raw state_dict), moves tensors to
    the backbone's device+dtype, and load_state_dict(strict=False) (the frozen-base
    keys are absent). Both the Advantage Server and Advantage Trainer call this at
    startup so they begin RL from identical warm weights (the both-warm invariant).
    """
    logger = get_logger()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first_trainable = next(p for p in backbone.parameters() if p.requires_grad)
    state_dict = {
        name: t.to(device=first_trainable.device, dtype=first_trainable.dtype)
        for name, t in payload["trainable_state_dict"].items()
    }
    result = backbone.load_state_dict(state_dict, strict=False)
    if len(result.unexpected_keys) > 0:
        logger.warning(
            f"load_warm_start: {len(result.unexpected_keys)} unexpected keys "
            f"(first few: {result.unexpected_keys[:3]}). Continuing."
        )
