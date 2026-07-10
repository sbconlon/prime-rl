"""Shared value-network training pieces, extracted from _train_step_action and
train() so the online Advantage Trainer and the offline warm-start regress V/Q+
with byte-identical forward/loss. These are the correctness-critical,
silent-failure-prone shared surface (the V-boundary read, the per-sample
weighting, the serialized backward); the outer loop differs per caller (online
Jin cycle vs offline epochs) and stays with each caller.
"""

from __future__ import annotations

import torch
from torch import optim

from prime_rl.advantage_trainer._prof import prof
from prime_rl.advantage_trainer.data import prepare_action_advantage_samples
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import AdvantageTrainingSample
from prime_rl.utils.logger import get_logger


def value_regression_backward(
    backbone: ValueNetworkBackbone,
    chunk: list[AdvantageTrainingSample],
    tokenizer,
    chunk_scale: float,
    device,
) -> tuple[float, float]:
    """One chunk: prepare, then the V leg and the Q+ leg (forward, per-sample-weighted
    MSE, serialized V-then-Q+ backward). Returns the (l_v, l_q) loss contributions
    already scaled by chunk_scale. Verbatim lift of _train_step_action's inner-chunk
    body: V read at each decision point's o-boundary from one all-positions forward;
    Q+ via forward_q_plus_action([o, a*]) per decision point.
    """
    with prof("advtrainer.prepare_batch_action", K=len(chunk)):
        prep = prepare_action_advantage_samples(chunk, tokenizer, device=device)
    D = int(prep.dp_sample_idx.numel())
    if D == 0:
        return 0.0, 0.0

    # --- V leg (batched all-positions forward, gather at boundaries) ---
    with prof("advtrainer.forward.v_action", sync_cuda=True):
        v_all, _ = backbone.forward_v_all_positions(prep.trajectory_input_ids)
    v_pred = v_all[prep.dp_sample_idx, prep.dp_v_pos]
    l_v = (prep.weights * (v_pred - prep.v_targets) ** 2).sum()
    with prof("advtrainer.backward.v_action", sync_cuda=True):
        (l_v * chunk_scale).backward()
    l_v_contrib = float(l_v.detach().item()) * chunk_scale
    del v_all, v_pred, l_v

    # --- Q+ leg: per decision point (right-padded obs would break the
    # append-and-read cat); per-dp backward keeps peak memory to one Q+ graph. ---
    l_q_contrib = 0.0
    for d in range(D):
        o_ids = torch.tensor([prep.q_plus_obs_ids[d]], dtype=torch.long, device=device)
        a_ids = torch.tensor([prep.q_plus_action_ids[d]], dtype=torch.long, device=device)
        q_pred = backbone.forward_q_plus_action(o_ids, a_ids)[0]
        w = float(prep.weights[d].item())
        sq = (q_pred - prep.q_plus_targets[d]) ** 2
        (sq * (w * chunk_scale)).backward()
        l_q_contrib += float(sq.detach().item()) * w * chunk_scale
    return l_v_contrib, l_q_contrib


def setup_value_optimizer(
    backbone: ValueNetworkBackbone, v_lr: float, q_plus_lr: float
) -> optim.Optimizer:
    """Per-slot AdamW: V (lora_*.0 + v_head) @ v_lr, Q+ (lora_*.1 + q_plus_head)
    @ q_plus_lr, V_target (lora_*.2 + v_target_head) @ lr=0. V_target is never
    gradient-trained (moved only by Polyak); lr=0 is explicit + defensive. Verbatim
    lift of the partition in advantage_trainer.train.train().
    """
    logger = get_logger()
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
            f"Found {len(other_trainable)} trainable params outside the "
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
    logger.info(f"Per-slot learning rates: V={v_lr:.2e}, Q+={q_plus_lr:.2e}, V_target=0.0")

    return optim.AdamW(
        [
            {"params": v_params, "lr": v_lr},
            {"params": q_plus_params, "lr": q_plus_lr},
            {"params": v_target_params, "lr": 0.0},
        ]
    )
