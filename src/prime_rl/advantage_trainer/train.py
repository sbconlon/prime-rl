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
from transformers import AutoModel, AutoTokenizer

from prime_rl.advantage_trainer.data import (
    prepare_advantage_sample,
    prepare_batched_advantage_samples,
)
from prime_rl.advantage_trainer._prof import prof, set_request_id
from prime_rl.advantage_trainer.ckpt import load_warm_start, setup_advantage_ckpt_manager
from prime_rl.advantage_trainer.shared import setup_value_optimizer, value_regression_backward
from prime_rl.utils.utils import resolve_latest_ckpt_step
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
    n_epochs: int = 1,
    minibatch_size: int | None = None,
    inner_batch_size: int = 1,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Run one Jin-aligned training cycle over a batch.

    Per-cycle structure (spec 20260516):
        for epoch in range(n_epochs):
            reshuffle batch.examples into minibatches of `minibatch_size`
            for mb in minibatches:
                optimizer.zero_grad()
                forward + backward over chunks of `inner_batch_size`
                  (grad accumulates inside the minibatch)
                optimizer.step()
        polyak_update_v_target(tau)

    Targets (v_targets, q_plus_targets) are computed once by the AdvServer
    under V_prev/Q+_prev and frozen for the entire cycle -- the AdvTrainer
    cannot recompute them, which structurally enforces the CFR+
    convergence invariant. Polyak update + weight broadcast fire ONCE per
    cycle, after the inner loop completes.

    Three-knob hierarchy:
        n_epochs       -- passes over the cycle's batch
        minibatch_size -- samples per optimizer step (SGD knob; Jin's MB_SIZE)
        inner_batch_size -- samples per forward+backward chunk (memory knob)

    `minibatch_size=None` means "use the full batch as one minibatch"
    (legacy behavior: one optimizer step per cycle). This recovers the
    pre-spec single-step behavior when paired with n_epochs=1; useful for
    parity tests and emergency rollback.

    Loss scaling: each chunk's loss is multiplied by K_chunk/mb_size_actual
    so the accumulated gradient is the mean-of-per-sample-means over the
    minibatch (matches Adam's expected scale; learning rate stays
    interpretable across minibatch sizes).

    The Polyak update fires AFTER the inner loop (so V_target tracks the
    fully-updated V) and BEFORE the weight broadcast (so the Server
    receives the latest V_target).
    """
    device = next(backbone.parameters()).device
    backbone.train()

    n_total = len(batch.examples)
    if n_total == 0:
        return {
            "loss": 0.0,
            "mean_loss": 0.0,
            "l_v": 0.0,
            "l_q": 0.0,
            "l_v_first_epoch": 0.0,
            "l_v_last_epoch": 0.0,
            "l_q_first_epoch": 0.0,
            "l_q_last_epoch": 0.0,
            "q_plus_target_abs_mean": 0.0,
            "inner_steps": 0,
            "n_samples": 0,
        }

    # Pre-compute a cycle-level diagnostic: |q_plus_target| averaged over
    # all valid positions across the batch. Spec's race-condition predicts
    # this stays bounded under the Jin-aligned loop instead of growing
    # 0.2 -> 0.9 as observed in the K=4 collapse run.
    q_plus_target_abs_sum = 0.0
    q_plus_target_count = 0
    if algorithm == "arm":
        for s in batch.examples:
            if s.q_plus_targets is None:
                continue
            for t in s.q_plus_targets:
                q_plus_target_abs_sum += abs(float(t))
                q_plus_target_count += 1
    q_plus_target_abs_mean = (
        q_plus_target_abs_sum / q_plus_target_count if q_plus_target_count else 0.0
    )

    # Resolve minibatch_size: None -> full batch (one optimizer step per epoch).
    mb_size = minibatch_size if minibatch_size is not None else n_total

    # Cycle-level accumulators.
    l_v_sum = 0.0
    l_q_sum = 0.0
    loss_sum = 0.0
    n_inner_steps = 0
    # Per-epoch accumulators -- the race diagnostic compares first-epoch
    # mean vs. last-epoch mean. Each epoch sweeps all n_total samples
    # (just shuffled differently), so the mean across an epoch's
    # minibatches is comparable across epochs apples-to-apples.
    # Comparing first-minibatch vs. last-minibatch directly would be
    # noisy because the shuffle makes them different sample subsets.
    first_epoch_l_v_sum = 0.0
    first_epoch_l_q_sum = 0.0
    first_epoch_steps = 0
    last_epoch_l_v_sum = 0.0
    last_epoch_l_q_sum = 0.0
    last_epoch_steps = 0

    for epoch in range(n_epochs):
        # Fresh shuffle each epoch (spec invariant: keeps consecutive
        # minibatch gradients directionally varied; reshuffling defeats
        # accidental correlation across epochs).
        if generator is not None:
            perm = torch.randperm(n_total, generator=generator).tolist()
        else:
            perm = torch.randperm(n_total).tolist()
        shuffled = [batch.examples[i] for i in perm]

        epoch_l_v = 0.0
        epoch_l_q = 0.0
        epoch_n_inner = 0

        # Walk minibatches; the last minibatch may be smaller than mb_size.
        for mb_start in range(0, n_total, mb_size):
            mb = shuffled[mb_start : mb_start + mb_size]
            mb_size_actual = len(mb)

            optimizer.zero_grad()
            mb_loss = 0.0
            mb_l_v = 0.0
            mb_l_q = 0.0

            # Memory chunking INSIDE the minibatch -- gradient accumulates,
            # one optimizer.step() applies the mb's mean-gradient.
            for chunk_start in range(0, mb_size_actual, inner_batch_size):
                chunk = mb[chunk_start : chunk_start + inner_batch_size]
                K = len(chunk)

                with prof("advtrainer.prepare_batch", K=K):
                    prepared = prepare_batched_advantage_samples(chunk, device=device)

                # Path B-simple: serialized V/Q+ backward. Each leg forwards,
                # computes its leg-specific loss, runs backward, and drops
                # its autograd graph before the next leg allocates. V's params
                # and Q+'s params are disjoint LoRA slots + disjoint heads, so
                # the two backwards produce identical gradients to the combined
                # backward -- see loss.py docstring. Peak activation memory is
                # max(V_graph, Q+_graph), not V_graph + Q+_graph.
                B_chunk, S = prepared.input_ids.shape
                mask_f = prepared.loss_mask.to(dtype=prepared.v_targets.dtype)
                valid_count = prepared.loss_mask.sum(dim=1).clamp(min=1)
                chunk_scale = float(K) / float(mb_size_actual)

                # --- V leg ---
                with prof("advtrainer.forward.v", sync_cuda=True, B=B_chunk, S=S):
                    v_pred, _ = backbone.forward_v_all_positions(prepared.input_ids)
                v_diff_sq = (v_pred - prepared.v_targets) ** 2
                l_v = ((v_diff_sq * mask_f).sum(dim=1) / valid_count).mean()
                with prof("advtrainer.backward.v", sync_cuda=True, K=K):
                    (l_v * chunk_scale).backward()
                l_v_value = l_v.detach().item()
                del v_pred, v_diff_sq, l_v

                # --- Q+ leg (ARM only) ---
                if algorithm == "arm":
                    if prepared.q_plus_targets is None:
                        raise ValueError(
                            "ARM requires q_plus_targets in the prepared batch."
                        )
                    with prof("advtrainer.forward.q_plus", sync_cuda=True, B=B_chunk, S=S):
                        q_plus_pred, _ = backbone.forward_q_plus_sampled_all_positions(
                            prepared.input_ids
                        )
                    q_diff_sq = (q_plus_pred - prepared.q_plus_targets) ** 2
                    l_q = ((q_diff_sq * mask_f).sum(dim=1) / valid_count).mean()
                    with prof("advtrainer.backward.q_plus", sync_cuda=True, K=K):
                        (l_q * chunk_scale).backward()
                    l_q_value = l_q.detach().item()
                    del q_plus_pred, q_diff_sq, l_q
                else:
                    l_q_value = 0.0

                mb_loss += (l_v_value + l_q_value) * chunk_scale
                mb_l_v += l_v_value * chunk_scale
                mb_l_q += l_q_value * chunk_scale

            with prof("advtrainer.optimizer_step", sync_cuda=True):
                optimizer.step()
            n_inner_steps += 1
            epoch_l_v += mb_l_v
            epoch_l_q += mb_l_q
            epoch_n_inner += 1
            l_v_sum += mb_l_v
            l_q_sum += mb_l_q
            loss_sum += mb_loss

        # End-of-epoch: stash first / last epoch's mean for the race
        # diagnostic. Each epoch's mean is the average loss over the full
        # batch's minibatches in this epoch's shuffle order; that's the
        # same n_total samples as every other epoch, so first- vs. last-
        # epoch means are apples-to-apples.
        if epoch == 0:
            first_epoch_l_v_sum = epoch_l_v
            first_epoch_l_q_sum = epoch_l_q
            first_epoch_steps = epoch_n_inner
        last_epoch_l_v_sum = epoch_l_v
        last_epoch_l_q_sum = epoch_l_q
        last_epoch_steps = epoch_n_inner

    with prof("advtrainer.polyak", sync_cuda=True):
        backbone.polyak_update_v_target(tau=polyak_tau)

    # Means across inner steps (the cycle-level summary), and per-epoch
    # first/last means for the within-cycle race diagnostic.
    l_v_mean = l_v_sum / n_inner_steps if n_inner_steps else 0.0
    l_q_mean = l_q_sum / n_inner_steps if n_inner_steps else 0.0
    loss_mean = loss_sum / n_inner_steps if n_inner_steps else 0.0
    l_v_first_epoch = first_epoch_l_v_sum / first_epoch_steps if first_epoch_steps else 0.0
    l_q_first_epoch = first_epoch_l_q_sum / first_epoch_steps if first_epoch_steps else 0.0
    l_v_last_epoch = last_epoch_l_v_sum / last_epoch_steps if last_epoch_steps else 0.0
    l_q_last_epoch = last_epoch_l_q_sum / last_epoch_steps if last_epoch_steps else 0.0

    return {
        # Backward-compat keys (existing log consumers expect these):
        "loss": float(loss_mean),
        "mean_loss": float(loss_mean),
        "l_v": float(l_v_mean),
        "l_q": float(l_q_mean),
        "n_samples": n_total,
        # New diagnostic keys: per-epoch mean compares full-batch sweeps,
        # not individual minibatches (which are non-comparable under
        # per-epoch reshuffling).
        "l_v_first_epoch": float(l_v_first_epoch),
        "l_v_last_epoch": float(l_v_last_epoch),
        "l_q_first_epoch": float(l_q_first_epoch),
        "l_q_last_epoch": float(l_q_last_epoch),
        "q_plus_target_abs_mean": float(q_plus_target_abs_mean),
        "inner_steps": int(n_inner_steps),
    }


def _train_step_action(
    backbone: ValueNetworkBackbone,
    optimizer: optim.Optimizer,
    batch: AdvantageTrainingBatch,
    tokenizer,
    polyak_tau: float,
    n_epochs: int = 1,
    minibatch_size: int | None = None,
    inner_batch_size: int = 1,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Action-level ARM training cycle: regress V/Q+ against the per-decision-point
    targets (AdvantageTrainingSample.decision_point_targets, Phase 6).

    Same Jin-aligned scaffolding as _train_step (epoch/minibatch reshuffle,
    zero_grad/step, serialized V-then-Q+ backward, chunk_scale, Polyak once after
    the inner loop) -- only the forward/loss change to per-decision-point:
      V:  forward_v_all_positions(trajectory) read at each o-boundary -> V(o_k).
      Q+: forward_q_plus_action([o_k, a*_k]) (Phase 5; o excludes the reasoning
          block, evaluated per decision point) -> Q+(o_k, a*_k).
    Loss is per-decision-point MSE with per-sample-equal weighting (weights from
    prepare_action_advantage_samples), so a many-turn rollout doesn't dominate.
    The weight broadcast stays in train() (as for the token-level path).
    """
    device = next(backbone.parameters()).device
    backbone.train()

    n_total = len(batch.examples)
    empty = {
        "loss": 0.0, "mean_loss": 0.0, "l_v": 0.0, "l_q": 0.0, "n_samples": 0,
        "l_v_first_epoch": 0.0, "l_v_last_epoch": 0.0,
        "l_q_first_epoch": 0.0, "l_q_last_epoch": 0.0,
        "q_plus_target_abs_mean": 0.0, "inner_steps": 0,
    }
    if n_total == 0:
        return empty

    q_abs_sum, q_abs_cnt = 0.0, 0
    for s in batch.examples:
        for dpt in (s.decision_point_targets or []):
            q_abs_sum += abs(float(dpt.q_plus_target))
            q_abs_cnt += 1
    q_plus_target_abs_mean = q_abs_sum / q_abs_cnt if q_abs_cnt else 0.0

    mb_size = minibatch_size if minibatch_size is not None else n_total
    l_v_sum = l_q_sum = loss_sum = 0.0
    n_inner_steps = 0
    first_lv = first_lq = first_steps = 0.0
    last_lv = last_lq = last_steps = 0.0

    for epoch in range(n_epochs):
        perm = (
            torch.randperm(n_total, generator=generator).tolist()
            if generator is not None
            else torch.randperm(n_total).tolist()
        )
        shuffled = [batch.examples[i] for i in perm]
        epoch_lv = epoch_lq = epoch_n = 0.0

        for mb_start in range(0, n_total, mb_size):
            mb = shuffled[mb_start : mb_start + mb_size]
            mb_size_actual = len(mb)
            optimizer.zero_grad()
            mb_l_v = mb_l_q = 0.0

            for chunk_start in range(0, mb_size_actual, inner_batch_size):
                chunk = mb[chunk_start : chunk_start + inner_batch_size]
                chunk_scale = float(len(chunk)) / float(mb_size_actual)
                l_v_contrib, l_q_contrib = value_regression_backward(
                    backbone, chunk, tokenizer, chunk_scale, device
                )
                mb_l_v += l_v_contrib
                mb_l_q += l_q_contrib

            optimizer.step()
            n_inner_steps += 1
            epoch_lv += mb_l_v
            epoch_lq += mb_l_q
            epoch_n += 1
            l_v_sum += mb_l_v
            l_q_sum += mb_l_q
            loss_sum += mb_l_v + mb_l_q

        if epoch == 0:
            first_lv, first_lq, first_steps = epoch_lv, epoch_lq, epoch_n
        last_lv, last_lq, last_steps = epoch_lv, epoch_lq, epoch_n

    backbone.polyak_update_v_target(tau=polyak_tau)

    return {
        "loss": float(loss_sum / n_inner_steps) if n_inner_steps else 0.0,
        "mean_loss": float(loss_sum / n_inner_steps) if n_inner_steps else 0.0,
        "l_v": float(l_v_sum / n_inner_steps) if n_inner_steps else 0.0,
        "l_q": float(l_q_sum / n_inner_steps) if n_inner_steps else 0.0,
        "n_samples": n_total,
        "l_v_first_epoch": float(first_lv / first_steps) if first_steps else 0.0,
        "l_v_last_epoch": float(last_lv / last_steps) if last_steps else 0.0,
        "l_q_first_epoch": float(first_lq / first_steps) if first_steps else 0.0,
        "l_q_last_epoch": float(last_lq / last_steps) if last_steps else 0.0,
        "q_plus_target_abs_mean": float(q_plus_target_abs_mean),
        "inner_steps": int(n_inner_steps),
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

    # Action-level ARM: tokenizer for the executed-action -> Q+ token-ids (shared
    # tokenize_action helper). Best-effort; the per-token (token-level/PPO) path
    # does not need it.
    tokenizer = None
    if config.algorithm == "arm":
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load tokenizer for action-level ARM: {exc}")

    v_lr = config.v_learning_rate if config.v_learning_rate is not None else config.learning_rate
    q_plus_lr = (
        config.q_plus_learning_rate if config.q_plus_learning_rate is not None else config.learning_rate
    )
    optimizer = setup_value_optimizer(backbone, v_lr, q_plus_lr)

    # Checkpoint manager. Mirrors the LLM trainer\'s setup_ckpt_managers
    # pattern: returns None when config.ckpt is None (no checkpointing
    # configured), in which case downstream save/load blocks no-op.
    ckpt_manager = setup_advantage_ckpt_manager(config.output_dir, config.ckpt)
    checkpoint_step: int | None = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    if checkpoint_step is not None and ckpt_manager is not None:
        loaded_step = ckpt_manager.load(checkpoint_step, backbone, optimizer)
        logger.info(
            f"Resuming AdvTrainer from checkpoint step {loaded_step} "
            f"(loaded from {ckpt_manager.get_ckpt_path(checkpoint_step)})"
        )
        resume_start_step = loaded_step
    elif config.warm_start_path:
        load_warm_start(backbone, config.warm_start_path)
        logger.info(f"Advantage Trainer: warm-starting value nets from {config.warm_start_path}")
        resume_start_step = 0
    else:
        resume_start_step = 0

    receiver = setup_advantage_training_batch_receiver(
        config.transport, input_dir=config.transport_input_dir
    )
    logger.info(
        f"Transport receiver ready ({config.transport.type}, "
        f"input_dir={config.transport_input_dir})"
    )

    step = resume_start_step
    while step < config.max_steps:
        # Wait for at least one batch.
        if not receiver.can_receive():
            time.sleep(0.1)
            continue
        with prof("advtrainer.recv"):
            batches = receiver.receive()

        for batch in batches:
            set_request_id(str(step))
            if torch.cuda.is_available():
                mem_start_gb = torch.cuda.memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                logger.info(f"step={step} cuda_allocated_start={mem_start_gb:.2f}GB")
            with prof("advtrainer.step.TOTAL", sync_cuda=True, step=step):
                with prof("advtrainer.train_step", sync_cuda=True):
                    # On this branch "arm" is action-level: regress per-decision-
                    # point targets via _train_step_action. The token-level
                    # _train_step stays for PPO (and as dead-code ARM regression).
                    if config.algorithm == "arm":
                        metrics = _train_step_action(
                            backbone=backbone,
                            optimizer=optimizer,
                            batch=batch,
                            tokenizer=tokenizer,
                            polyak_tau=config.polyak_tau,
                            n_epochs=config.n_epochs,
                            minibatch_size=config.minibatch_size,
                            inner_batch_size=config.inner_batch_size,
                        )
                    else:
                        metrics = _train_step(
                            backbone=backbone,
                            optimizer=optimizer,
                            batch=batch,
                            algorithm=config.algorithm,
                            polyak_tau=config.polyak_tau,
                            n_epochs=config.n_epochs,
                            minibatch_size=config.minibatch_size,
                            inner_batch_size=config.inner_batch_size,
                        )

                # Phase 7c: broadcast updated weights to the Advantage Server
                # (best-effort; failures logged but non-fatal).
                if config.advantage_server_url and metrics["n_samples"] > 0:
                    with prof("advtrainer.broadcast", sync_cuda=True):
                        broadcast_ok = _broadcast_weights(
                            backbone, config.advantage_server_url
                        )
                    if not broadcast_ok:
                        logger.warning(f"step={step}: weight broadcast failed")

                # Race-condition diagnostic on every cycle line: first/last
                # epoch mean L_V/L_Q captures whether the inner loop is
                # converging the value networks within a cycle (spec 20260516);
                # q_plus_target_abs_mean tracks the predicted-bounded-vs-
                # growing CFR+ target invariant.
                logger.info(
                    f"step={step} inner_steps={metrics['inner_steps']} "
                    f"l_v_first_epoch={metrics['l_v_first_epoch']:.4f} "
                    f"l_v_last_epoch={metrics['l_v_last_epoch']:.4f} "
                    f"l_v_mean={metrics['l_v']:.4f} "
                    f"l_q_first_epoch={metrics['l_q_first_epoch']:.4f} "
                    f"l_q_last_epoch={metrics['l_q_last_epoch']:.4f} "
                    f"l_q_mean={metrics['l_q']:.4f} "
                    f"q_tgt_abs={metrics['q_plus_target_abs_mean']:.4f} "
                    f"n_samples={metrics['n_samples']}"
                )
                step += 1

                # Save checkpoint at interval (mirrors LLM trainer\'s save block).
                if (
                    ckpt_manager is not None
                    and config.ckpt is not None
                    and config.ckpt.interval is not None
                    and step % config.ckpt.interval == 0
                    and step < config.max_steps
                ):
                    with prof("advtrainer.ckpt.save", step=step):
                        logger.info(f"Saving AdvTrainer checkpoint at step {step}")
                        ckpt_manager.save(step, backbone, optimizer)
                        ckpt_manager.maybe_clean()

            if torch.cuda.is_available():
                mem_peak_gb = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"step={step - 1} cuda_peak={mem_peak_gb:.2f}GB")

            if step >= config.max_steps:
                break

    # Final checkpoint at end of training (mirrors LLM trainer is_last_step).
    if (
        ckpt_manager is not None
        and config.ckpt is not None
        and config.ckpt.interval is not None
    ):
        logger.info(f"Saving final AdvTrainer checkpoint at step {step}")
        ckpt_manager.save(step, backbone, optimizer)
        ckpt_manager.maybe_clean()

    logger.success(f"Advantage Trainer finished after {step} steps.")
    receiver.close()


def main() -> None:
    """Main entry point. Run via `uv run advantage-trainer @ <config.toml>`."""
    config = cli(AdvantageTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
