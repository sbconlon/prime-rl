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

import io
import time
from typing import Literal

import httpx
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
# Weight broadcast (Trainer -> Advantage Server, HTTP POST per step)
# ---------------------------------------------------------------------------


def _serialize_trainable_state_dict(backbone: ValueNetworkBackbone) -> bytes:
    """Serialize only the trainable parameters (LoRA + value heads) via torch.save.

    Returns the raw bytes ready to POST to /update_weights. The Advantage
    Server's load_state_dict(strict=False) tolerates the missing frozen-base
    keys.
    """
    state_dict = {
        name: param.detach().cpu()
        for name, param in backbone.named_parameters()
        if param.requires_grad
    }
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def _broadcast_weights(
    backbone: ValueNetworkBackbone, server_url: str, *, timeout: float = 60.0
) -> bool:
    """Best-effort broadcast: serialize the trainable params and POST them
    to the Advantage Server. Returns True on success, False on any error.

    Errors are non-fatal -- the Trainer continues, the Server runs one
    step with stale weights. Repeated failures suggest a real connectivity
    problem and the operator should investigate.
    """
    try:
        body = _serialize_trainable_state_dict(backbone)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{server_url.rstrip('/')}/update_weights",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        if response.status_code != 200:
            get_logger().warning(
                "weight broadcast: server responded %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False
        return True
    except Exception as exc:
        get_logger().warning("weight broadcast failed: %s", exc)
        return False


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

    # Per-sample backward (Phase 10 perf fix): accumulating per-sample losses
    # into one autograd graph and backwarding at the end retains every
    # sample's activations until the final backward, which OOMs at batch
    # sizes typical of the canary (the strict-causal Q+ mask from §4.1
    # kicks SDPA onto its math backend, making each sample's saved
    # activations ~GB-scale). Backward per sample releases the graph
    # immediately; gradients accumulate in-place on the parameters, then we
    # scale by 1/n_samples to recover mean-of-losses semantics.
    optimizer.zero_grad()
    total_loss_scalar = 0.0
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

        out.loss.backward()  # graph freed after each sample
        total_loss_scalar += out.loss.detach().item()
        total_l_v += out.metrics["l_v"].item()
        total_l_q += out.metrics["l_q"].item()
        n_samples += 1

    if n_samples == 0:
        # Empty batch -- nothing to do.
        return {"loss": 0.0, "mean_loss": 0.0, "l_v": 0.0, "l_q": 0.0, "n_samples": 0}

    # Scale accumulated gradients to match the mean-of-losses semantics of
    # the previous (loss-accumulation + single backward) implementation.
    for p_ in backbone.parameters():
        if p_.grad is not None:
            p_.grad.div_(n_samples)

    optimizer.step()
    backbone.polyak_update_v_target(tau=polyak_tau)

    mean_loss_scalar = total_loss_scalar / n_samples
    return {
        "loss": float(total_loss_scalar),
        "mean_loss": float(mean_loss_scalar),
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

    # Match the AdvSrv's CUDA + BF16 placement (server.py). FP32 on a
    # 0.6B model with the Phase 10 batch shape was the second contributor
    # to the OOM (the first being the per-sample backward fix above).
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    logger.info(f"Loading value backbone (device={device}, dtype={dtype})")
    base_model = AutoModel.from_pretrained(config.model.base_model_name, dtype=dtype)
    backbone = ValueNetworkBackbone(
        base_model,
        lora_config=config.model.lora,
        polyak_tau=config.polyak_tau,
    )
    backbone = backbone.to(device=device, dtype=dtype)
    logger.success("Value backbone loaded")

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate)

    receiver = setup_advantage_training_batch_receiver(
        config.transport, input_dir=config.transport_input_dir
    )
    logger.info(
        f"Transport receiver ready ({config.transport.type}, "
        f"input_dir={config.transport_input_dir})"
    )

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

            # Phase 7c: broadcast updated weights to the Advantage Server
            # (best-effort; failures logged but non-fatal).
            if config.advantage_server_url and metrics["n_samples"] > 0:
                broadcast_ok = _broadcast_weights(
                    backbone, config.advantage_server_url
                )
                if not broadcast_ok:
                    logger.warning(f"step={step}: weight broadcast failed")

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
