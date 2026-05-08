"""Phase 7b: Advantage Trainer process entrypoint.

Standalone executable that mirrors `prime_rl.trainer.rl.train.train` but
swaps in:
  - Model: ValueNetworkBackbone (V/V_target/Q+ as 3 LoRA slots on a
    frozen SFT'd Qwen2.5-1.5B base).
  - Forward: value_forward (Phase 6.5 optimized all-positions methods).
  - Loss: value_regression_loss_fn (MSE on V and optionally Q+, masked
    to completion-mask=True positions).
  - Polyak update of V_target after each gradient step.
  - Transport: AdvantageTrainingBatch (Phase 7b), not TrainingBatch.

What's intentionally NOT here in 7b (deferred to 7c / Phase 9-10 prep):
  - Weight broadcast to the Advantage Server (stub: log and skip).
  - Checkpoint / resume.
  - Wandb / metrics-server / heartbeat.
  - FSDP wrapping (single-GPU is the committed default per phase doc DQ2).
  - Multi-batch packing / micro-batching (one sample at a time per step).

The `_train_step` function is the testable unit -- given a fully-loaded
backbone, optimizer, and an AdvantageTrainingBatch, it runs one step and
returns metrics. The `train` entrypoint loops `_train_step` against the
real transport receiver. Tests exercise `_train_step` directly with
fabricated batches; the four-process pipeline integration test (cluster)
exercises `train` end-to-end.
"""

from __future__ import annotations

import time
from typing import Literal

import torch
from torch import optim
from transformers import AutoModel

from prime_rl.advantage_trainer.data import prepare_advantage_sample
from prime_rl.advantage_trainer.forward import value_forward
from prime_rl.advantage_trainer.loss import (
    ValueLossInputs,
    ValueLossOutputs,
    value_regression_loss_fn,
)
from prime_rl.configs.advantage_trainer import AdvantageTrainerConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.advantage_batch import (
    setup_advantage_training_batch_receiver,
)
from prime_rl.transport.types import AdvantageTrainingBatch
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger


# ---------------------------------------------------------------------------
# Per-step logic (testable in isolation -- no transport, no model loading)
# ---------------------------------------------------------------------------


def _train_step(
    backbone: ValueNetworkBackbone,
    optimizer: optim.Optimizer,
    batch: AdvantageTrainingBatch,
    algorithm: Literal["ppo", "arm"],
    polyak_tau: float,
) -> dict[str, float]:
    """Run one training step over a batch.

    For each AdvantageTrainingSample in the batch:
      1. Prepare tensors (input_ids + aligned targets + loss_mask) via the
         data adapter.
      2. Forward through V (and Q+ if ARM) via Phase 6.5 all-positions methods.
      3. Compute MSE regression loss on completion-mask=True positions.
      4. Accumulate the loss across samples; one backward pass per batch.
      5. Optimizer step.
      6. Polyak update of V_target's adapter slot.

    Returns aggregated metrics: total loss, per-sample mean loss, l_v, l_q.

    The Polyak update fires AFTER the optimizer step (so V_target tracks the
    just-updated V) and BEFORE any weight broadcast (so the Server receives
    the latest V_target). Phase 7c wires the broadcast.
    """
    device = next(backbone.parameters()).device
    backbone.train()

    # Sum loss across samples; one backward pass for the whole batch.
    total_loss: torch.Tensor | None = None
    total_l_v = 0.0
    total_l_q = 0.0
    n_samples = 0

    for sample in batch.examples:
        prepared = prepare_advantage_sample(sample, device=device)

        v_pred, q_plus_pred = value_forward(
            backbone, prepared.input_ids, algorithm=algorithm
        )

        loss_inputs = ValueLossInputs(
            v_predictions=v_pred[0],
            v_targets=prepared.v_targets,
            q_plus_predictions=q_plus_pred[0] if q_plus_pred is not None else None,
            q_plus_targets=prepared.q_plus_targets,
            loss_mask=prepared.loss_mask,
            algorithm=algorithm,
        )
        out: ValueLossOutputs = value_regression_loss_fn(loss_inputs)

        # Accumulate. Cloning isn't needed since each sample's loss is a
        # fresh tensor with its own graph -- summation chains the graphs.
        if total_loss is None:
            total_loss = out.loss
        else:
            total_loss = total_loss + out.loss
        total_l_v += out.metrics["l_v"].item()
        total_l_q += out.metrics["l_q"].item()
        n_samples += 1

    if total_loss is None or n_samples == 0:
        # Empty batch -- nothing to do.
        return {"loss": 0.0, "mean_loss": 0.0, "l_v": 0.0, "l_q": 0.0, "n_samples": 0}

    mean_loss = total_loss / n_samples

    optimizer.zero_grad()
    mean_loss.backward()
    optimizer.step()
    backbone.polyak_update_v_target(tau=polyak_tau)

    return {
        "loss": float(total_loss.detach().item()),
        "mean_loss": float(mean_loss.detach().item()),
        "l_v": total_l_v / n_samples,
        "l_q": total_l_q / n_samples,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Process entrypoint
# ---------------------------------------------------------------------------


def train(config: AdvantageTrainerConfig) -> None:
    """Advantage Trainer training loop.

    Loads the value backbone, sets up the optimizer and transport receiver,
    then loops over received batches.
    """
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "advantage_trainer.log"
        if config.log.file
        else None,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting Advantage Trainer (algorithm={config.algorithm})")
    logger.info(f"Loading value backbone from {config.model.base_model_name}")

    base_model = AutoModel.from_pretrained(config.model.base_model_name)
    backbone = ValueNetworkBackbone(
        base_model,
        lora_config=config.model.lora,
        polyak_tau=config.polyak_tau,
    )
    if torch.cuda.is_available():
        backbone = backbone.to("cuda")
    logger.success("Value backbone loaded")

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate)

    receiver = setup_advantage_training_batch_receiver(config.transport)
    logger.info(f"Transport receiver ready ({config.transport.type})")

    step = 0
    while step < config.max_steps:
        # Wait for at least one batch.
        if not receiver.can_receive():
            time.sleep(0.1)
            continue
        batches = receiver.receive()

        for batch in batches:
            metrics = _train_step(
                backbone=backbone,
                optimizer=optimizer,
                batch=batch,
                algorithm=config.algorithm,
                polyak_tau=config.polyak_tau,
            )
            logger.info(
                f"step={step} mean_loss={metrics['mean_loss']:.4f} "
                f"l_v={metrics['l_v']:.4f} l_q={metrics['l_q']:.4f} "
                f"n_samples={metrics['n_samples']}"
            )
            step += 1
            if step >= config.max_steps:
                break

    logger.success(f"Advantage Trainer finished after {step} steps.")
    receiver.close()


def main() -> None:
    """Main entry point. Run via `uv run advantage-trainer @ <config.toml>`."""
    config = cli(AdvantageTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
