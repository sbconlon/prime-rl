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
    # Per-leg weights (branched dataset D-10): equal to prep.weights unless a
    # target masks a leg (v_weight / q_weight = 0).
    l_v = (prep.v_weights * (v_pred - prep.v_targets) ** 2).sum()
    with prof("advtrainer.backward.v_action", sync_cuda=True):
        (l_v * chunk_scale).backward()
    l_v_contrib = float(l_v.detach().item()) * chunk_scale
    del v_all, v_pred, l_v

    # --- Q+ leg: per decision point (right-padded obs would break the
    # append-and-read cat); per-dp backward keeps peak memory to one Q+ graph. ---
    l_q_contrib = 0.0
    # One timer for the whole Q+ leg (D forward + backward passes); it was the
    # unprofiled majority of a chunk's time.
    with prof("advtrainer.q_plus_action", sync_cuda=True, D=D):
        for d in range(D):
            o_ids = torch.tensor([prep.q_plus_obs_ids[d]], dtype=torch.long, device=device)
            a_ids = torch.tensor([prep.q_plus_action_ids[d]], dtype=torch.long, device=device)
            q_pred = backbone.forward_q_plus_action(o_ids, a_ids)[0]
            w = float(prep.q_weights[d].item())
            sq = (q_pred - prep.q_plus_targets[d]) ** 2
            (sq * (w * chunk_scale)).backward()
            l_q_contrib += float(sq.detach().item()) * w * chunk_scale
    return l_v_contrib, l_q_contrib


class MasterWeightAdamW:
    """AdamW with float32 master weights and float32 state for parameters that
    live in a lower precision (bfloat16 on GPU).

    Why: with bfloat16 parameters torch's AdamW keeps its state in bfloat16.
    The second-moment decay (beta2 = 0.999, a 0.1 % change per step) is below
    bfloat16's resolution, so it rounds away and the estimate can only grow: a
    gradient spike then lowers that parameter's learning rate permanently.
    Measured on the branched warm-start (job 755044): 0 of 18.5M second-moment
    entries per network decreased over 100 steps. Small weight updates are
    quantised the same way.

    Each trainable parameter gets a float32 master. step() copies the gradients
    to the masters, clips per group there, runs AdamW, and rounds the masters
    into the model's parameters. A group whose learning rate is 0 and a
    parameter that received no gradient are left alone, so V_target (moved only
    by Polyak updates on the model) is never overwritten. After loading weights
    into the model call sync_from_params().

    Duck-types the torch optimizer surface the trainers and the checkpoint
    manager use: param_groups, zero_grad, step, state_dict, load_state_dict."""

    def __init__(self, groups: list[dict]):
        self._model_params: list[list] = [list(g["params"]) for g in groups]
        self._masters: list[list] = [
            [torch.nn.Parameter(p_.detach().to(torch.float32).clone()) for p_ in ps]
            for ps in self._model_params
        ]
        self._opt = optim.AdamW(
            [{"params": ms, "lr": g["lr"]} for ms, g in zip(self._masters, groups)]
        )

    @property
    def param_groups(self) -> list[dict]:
        return self._opt.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for ps in self._model_params:
            for p_ in ps:
                p_.grad = None
        self._opt.zero_grad(set_to_none=True)

    def step(self, max_grad_norm: float | None = None) -> list[float]:
        """One update. Returns each group's gradient norm before clipping
        (float32). max_grad_norm clips each group separately; None measures only."""
        clip = float("inf") if max_grad_norm is None else float(max_grad_norm)
        norms: list[float] = []
        touched: list[tuple] = []
        for gi, (ps, ms) in enumerate(zip(self._model_params, self._masters)):
            if self._opt.param_groups[gi]["lr"] == 0.0:
                # A frozen group (V_target): its parameters are moved on the model
                # by Polyak updates, and the multi-slot LoRA hands them zero-valued
                # (not None) gradients. Never step it and never write its stale
                # masters back over the model.
                for m_ in ms:
                    m_.grad = None
                norms.append(0.0)
                continue
            with_grad = []
            for p_, m_ in zip(ps, ms):
                if p_.grad is None:
                    m_.grad = None
                    continue
                m_.grad = p_.grad.detach().to(torch.float32)
                with_grad.append(m_)
                touched.append((p_, m_))
            norms.append(float(torch.nn.utils.clip_grad_norm_(with_grad, clip)) if with_grad else 0.0)
        self._opt.step()
        with torch.no_grad():
            for p_, m_ in touched:
                p_.copy_(m_)  # rounds the float32 master to the parameter's dtype
        return norms

    def coherence(self) -> list[float]:
        """Mean |m_hat| / sqrt(v_hat) per group, from the bias-corrected AdamW
        moments: about 1 when the gradient keeps its sign from step to step
        (consistent signal), about 0.23 for random signs (pure noise, beta1 = 0.9),
        about 0.05 when the sign flips every step (the update overshoots). It is
        the size of the actual update relative to lr. 0.0 for a group without state."""
        out: list[float] = []
        for group, ms in zip(self._opt.param_groups, self._masters):
            beta1, beta2 = group["betas"]
            total = 0.0
            count = 0
            for m_ in ms:
                st = self._opt.state.get(m_)
                if not st or "exp_avg_sq" not in st:
                    continue
                n = float(st["step"])
                if n <= 0:
                    continue
                m_hat = st["exp_avg"].abs() / (1.0 - beta1**n)
                v_hat = st["exp_avg_sq"] / (1.0 - beta2**n)
                r = m_hat / (v_hat.sqrt() + 1e-12)
                total += float(r.sum())
                count += r.numel()
            out.append(total / count if count else 0.0)
        return out

    def sync_from_params(self) -> None:
        """Reset the masters to the model's current parameters (after a load)."""
        with torch.no_grad():
            for ps, ms in zip(self._model_params, self._masters):
                for p_, m_ in zip(ps, ms):
                    m_.copy_(p_.detach().to(torch.float32))

    def state_dict(self) -> dict:
        return self._opt.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        # torch casts loaded state tensors to the (float32) master dtype, so a
        # checkpoint written by the bfloat16 optimizer loads too.
        self._opt.load_state_dict(state_dict)


def setup_value_optimizer(
    backbone: ValueNetworkBackbone,
    v_lr: float,
    q_plus_lr: float,
    fp32_master: bool = False,
    head_lr: float | None = None,
) -> "optim.Optimizer | MasterWeightAdamW":
    """Per-slot AdamW: V (lora_*.0 + v_head) @ v_lr, Q+ (lora_*.1 + q_plus_head)
    @ q_plus_lr, V_target (lora_*.2 + v_target_head) @ lr=0. V_target is never
    gradient-trained (moved only by Polyak); lr=0 is explicit + defensive. Verbatim
    lift of the partition in advantage_trainer.train.train().

    head_lr (optional): v_head and q_plus_head leave their slot's group and get
    two groups of their own, appended as groups 3 and 4, at head_lr. Why: the
    heads read the base model's final hidden state, whose L1 norm is about 4,000
    (final-norm gain: L2 137, L1 5,315 on the 1.5B SFT base), so one Adam step at
    rate r moves a prediction by up to ~4,000 r. At the LoRA's rate (3e-4) that
    exceeds the 0/1 target range: measured on job 756302, the head gradient
    flipped sign every step (mean |m|/sqrt(v) 0.04 for V's head) and the head held
    98 % (V) / 66 % (Q+) of the slot's gradient energy, so slot-level clipping
    scaled the LoRA's gradient by a factor that changed 40-100x between steps and
    froze it. Separate groups give the heads their own rate and their own clip.
    """
    logger = get_logger()
    v_params: list = []
    q_plus_params: list = []
    v_target_params: list = []
    other_trainable: list[tuple[str, "torch.nn.Parameter"]] = []
    v_head_params: list = []
    q_head_params: list = []
    for name, p_ in backbone.named_parameters():
        if not p_.requires_grad:
            continue
        if head_lr is not None and "v_head" in name and "v_target_head" not in name:
            v_head_params.append(p_)
            continue
        if head_lr is not None and "q_plus_head" in name:
            q_head_params.append(p_)
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

    n_trainable = sum(
        p_.numel() for p_ in v_params + q_plus_params + v_target_params + v_head_params + q_head_params
    )
    logger.info(
        f"Trainable parameters: {n_trainable:,} "
        f"(V={sum(p_.numel() for p_ in v_params):,}, "
        f"Q+={sum(p_.numel() for p_ in q_plus_params):,}, "
        f"V_target={sum(p_.numel() for p_ in v_target_params):,})"
    )
    logger.info(f"Per-slot learning rates: V={v_lr:.2e}, Q+={q_plus_lr:.2e}, V_target=0.0")

    groups = [
        {"params": v_params, "lr": v_lr},
        {"params": q_plus_params, "lr": q_plus_lr},
        {"params": v_target_params, "lr": 0.0},
    ]
    if head_lr is not None:
        if not v_head_params or not q_head_params:
            raise ValueError("head_lr was given but v_head / q_plus_head parameters were not found")
        groups.append({"params": v_head_params, "lr": head_lr})
        groups.append({"params": q_head_params, "lr": head_lr})
        logger.info(f"Value heads in their own optimizer groups at lr={head_lr:.2e}")
    if fp32_master:
        return MasterWeightAdamW(groups)
    return optim.AdamW(groups)
