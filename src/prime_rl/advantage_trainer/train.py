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

from prime_rl.advantage_trainer.data import (
    prepare_advantage_sample,
    prepare_batched_advantage_samples,
)
from prime_rl.advantage_trainer.forward import value_forward
from prime_rl.advantage_trainer.loss import (
    BatchedValueLossInputs,
    ValueLossInputs,
    ValueLossOutputs,
    value_regression_loss_fn,
    value_regression_loss_fn_batched,
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
    inner_batch_size: int = 1,
) -> dict[str, float]:
    """Run one training step over a batch.

    Micro-batches the N samples into chunks of `inner_batch_size`. For
    each chunk:
      1. Right-pad K samples into [K, S_max] tensors via the batched
         data adapter.
      2. One forward through V (and Q+ if ARM) over [K, S_max].
      3. Vectorized MSE regression loss with per-sample-mean semantics
         (so long rollouts don't dominate gradient).
      4. Scaled backward (chunk_loss * K_chunk / n_total) so summing
         across chunks recovers the original mean-of-per-sample-mean.
    Then one optimizer step + Polyak update of V_target.

    inner_batch_size=1 recovers the pre-lever-2 per-sample backward
    behavior (one Python forward per sample). inner_batch_size=8 (the
    default in config) amortizes Python overhead and improves GPU
    utilization on the projection matmuls.

    The Polyak update fires AFTER the optimizer step (so V_target tracks
    the just-updated V) and BEFORE any weight broadcast (so the Server
    receives the latest V_target).
    """
    device = next(backbone.parameters()).device
    backbone.train()

    n_total = len(batch.examples)
    if n_total == 0:
        return {"loss": 0.0, "mean_loss": 0.0, "l_v": 0.0, "l_q": 0.0, "n_samples": 0}

    optimizer.zero_grad()
    total_loss_scalar = 0.0
    total_l_v = 0.0
    total_l_q = 0.0

    # Iterate chunks of size inner_batch_size (last chunk may be smaller).
    for start in range(0, n_total, inner_batch_size):
        chunk = batch.examples[start : start + inner_batch_size]
        K = len(chunk)

        prepared = prepare_batched_advantage_samples(chunk, device=device)

        v_pred, q_plus_pred = value_forward(
            backbone, prepared.input_ids, algorithm=algorithm
        )

        loss_inputs = BatchedValueLossInputs(
            v_predictions=v_pred,
            v_targets=prepared.v_targets,
            q_plus_predictions=q_plus_pred if q_plus_pred is not None else None,
            q_plus_targets=prepared.q_plus_targets,
            loss_mask=prepared.loss_mask,
            algorithm=algorithm,
        )
        out: ValueLossOutputs = value_regression_loss_fn_batched(loss_inputs)

        # Scale by K/n_total before backward so summing across chunks
        # recovers mean-of-per-sample-mean. (Each chunk_loss is already a
        # mean over its K samples; multiplying by K gives sum-over-samples;
        # dividing by n_total at the end of summation gives the overall
        # mean-of-per-sample.)
        chunk_scale = float(K) / float(n_total)
        (out.loss * chunk_scale).backward()

        # Track unscaled per-chunk metrics, weight by K/n_total for the
        # reported mean across the whole batch.
        total_loss_scalar += out.loss.detach().item() * chunk_scale
        total_l_v += out.metrics["l_v"].item() * chunk_scale
        total_l_q += out.metrics["l_q"].item() * chunk_scale

    optimizer.step()
    backbone.polyak_update_v_target(tau=polyak_tau)

    # mean_loss is the same as total_loss_scalar here -- both already
    # weighted by K/n_total. Kept as two fields for backward-compat with
    # the previous return schema and downstream logging.
    return {
        "loss": float(total_loss_scalar),
        "mean_loss": float(total_loss_scalar),
        "l_v": total_l_v,
        "l_q": total_l_q,
        "n_samples": n_total,
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

    # Phase 10 LR decoupling: split trainable params into V / Q+ / V_target
    # groups so the optimizer can apply per-slot learning rates. V_target
    # params get gradient = 0 (no loss flows through them) and are
    # overwritten by polyak_update_v_target after optimizer.step, so
    # putting them in an lr=0 group is just explicit + defensive.
    v_params: list = []
    q_plus_params: list = []
    v_target_params: list = []
    other_trainable: list[tuple[str, "torch.nn.Parameter"]] = []
    for name, p_ in backbone.named_parameters():
        if not p_.requires_grad:
            continue
        if name.endswith(".lora_A.0") or name.endswith(".lora_B.0") or (
            "v_head" in name and "v_target_head" not in name
        ):
            v_params.append(p_)
        elif name.endswith(".lora_A.1") or name.endswith(".lora_B.1") or "q_plus_head" in name:
            q_plus_params.append(p_)
        elif name.endswith(".lora_A.2") or name.endswith(".lora_B.2") or "v_target_head" in name:
            v_target_params.append(p_)
        else:
            other_trainable.append((name, p_))

    if other_trainable:
        logger.warning(
            f"AdvTrainer found {len(other_trainable)} trainable params outside the "
            f"V/Q+/V_target partition; assigning them to the V group as a "
            f"conservative default. First few names: "
            f"{[name for name, _ in other_trainable[:5]]}"
        )
        v_params.extend(p_ for _, p_ in other_trainable)

    n_trainable = sum(p_.numel() for p_ in v_params + q_plus_params + v_target_params)
    logger.info(
        f"Trainable parameters: {n_trainable:,} "
        f"(V={sum(p_.numel() for p_ in v_params):,}, "
        f"Q+={sum(p_.numel() for p_ in q_plus_params):,}, "
        f"V_target={sum(p_.numel() for p_ in v_target_params):,})"
    )

    v_lr = config.v_learning_rate if config.v_learning_rate is not None else config.learning_rate
    q_plus_lr = (
        config.q_plus_learning_rate if config.q_plus_learning_rate is not None else config.learning_rate
    )
    logger.info(f"Per-slot learning rates: V={v_lr:.2e}, Q+={q_plus_lr:.2e}, V_target=0.0")

    optimizer = optim.AdamW(
        [
            {"params": v_params, "lr": v_lr},
            {"params": q_plus_params, "lr": q_plus_lr},
            {"params": v_target_params, "lr": 0.0},
        ]
    )

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
                inner_batch_size=config.inner_batch_size,
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
