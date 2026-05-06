"""Advantage Server compute logic.

Given a `list[TrainingSample]` from a single rollout (post-`interleave_rollout`,
each with `advantages=None`), construct the per-token V/V_target/Q+ tensors by
walking each completion-mask=True position and calling the naive
`ValueNetworkBackbone` forward methods, then dispatch to Phase 2's PPO GAE or
Phase 3's ARM regret-matching to produce per-token advantages and regression
targets. Returns paired `(TrainingSample, AdvantageTrainingSample)` outputs.

Phase 6 = correctness only. The naive forward path is one full LLM forward
per V/Q+/V_target call, which is unusably slow at production rollout length.
Phase 6.5 layers KV-cached / batched-K-candidate optimizations underneath this
compute path without changing its interface.

Sample-type note: Phase 6's wire type is `TrainingSample` (the canonical
sample type from Phase 1); the master plan's `LLMTrainingSample` rename is a
deferred edit pass after Phase 6 is locked. The compute layer treats them
identically.
"""

from __future__ import annotations

from typing import Literal

import torch

from prime_rl.orchestrator.per_token_advantage import (
    ArmAdvantageInputs,
    PerTokenAdvantageOutputs,
    PpoAdvantageInputs,
    arm_regret_matching_advantage_fn,
    ppo_gae_advantage_fn,
)
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample


def compute_advantages_and_targets_ppo(
    samples: list[TrainingSample],
    episodic_reward: float,
    is_terminal: bool,
    backbone: ValueNetworkBackbone,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
    """Compute PPO GAE per-token advantages + v_targets for one rollout.

    Walks the joint trajectory view (concatenated mask=True positions across
    all `samples`), calls `backbone.forward_v` and `backbone.forward_v_target`
    once per active position to build `v_all` and `v_target_all`, then
    dispatches to `ppo_gae_advantage_fn` (Phase 2). The output is splayed
    back into per-completion-token arrays and packaged into paired outputs.
    """
    v_all, v_target_all, _, _ = _build_per_token_value_tensors(
        samples=samples, backbone=backbone, need_q_plus=False
    )
    inputs = PpoAdvantageInputs(
        samples=samples,
        episodic_reward=episodic_reward,
        is_terminal=is_terminal,
        v_all=v_all,
        v_target_all=v_target_all,
        gamma=gamma,
        lam=lam,
    )
    out = ppo_gae_advantage_fn(inputs)
    return _build_paired_outputs(samples, out)


def compute_advantages_and_targets_arm(
    samples: list[TrainingSample],
    episodic_reward: float,
    is_terminal: bool,
    backbone: ValueNetworkBackbone,
    *,
    gamma: float = 0.99,
    n_step: int = 5,
) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
    """Compute ARM regret-matching per-token advantages + v_targets + q_plus_targets
    for one rollout.

    Same structure as PPO, plus per-position Q+ at the sampled action and at
    each of K=top_k_action_set_size candidates. Requires
    `samples[i].completion_top_k_token_ids` to be populated (Phase 5 toggle on).
    """
    v_all, v_target_all, q_plus_sampled_all, q_plus_candidates = _build_per_token_value_tensors(
        samples=samples, backbone=backbone, need_q_plus=True
    )
    assert q_plus_sampled_all is not None and q_plus_candidates is not None
    inputs = ArmAdvantageInputs(
        samples=samples,
        episodic_reward=episodic_reward,
        is_terminal=is_terminal,
        v_all=v_all,
        v_target_all=v_target_all,
        q_plus_sampled_all=q_plus_sampled_all,
        q_plus_candidates=q_plus_candidates,
        gamma=gamma,
        n_step=n_step,
    )
    out = arm_regret_matching_advantage_fn(inputs)
    return _build_paired_outputs(samples, out)


def compute_advantages_and_targets(
    samples: list[TrainingSample],
    episodic_reward: float,
    is_terminal: bool,
    algorithm: Literal["ppo", "arm"],
    backbone: ValueNetworkBackbone,
    **kwargs: float | int,
) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
    """Dispatch to the per-algorithm compute path."""
    if algorithm == "ppo":
        return compute_advantages_and_targets_ppo(
            samples=samples,
            episodic_reward=episodic_reward,
            is_terminal=is_terminal,
            backbone=backbone,
            **kwargs,
        )
    if algorithm == "arm":
        return compute_advantages_and_targets_arm(
            samples=samples,
            episodic_reward=episodic_reward,
            is_terminal=is_terminal,
            backbone=backbone,
            **kwargs,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_per_token_value_tensors(
    samples: list[TrainingSample],
    backbone: ValueNetworkBackbone,
    need_q_plus: bool,
) -> tuple[
    torch.Tensor,                  # v_all,           shape [N_active]
    torch.Tensor,                  # v_target_all,    shape [N_active]
    torch.Tensor | None,           # q_plus_sampled_all, shape [N_active] or None
    torch.Tensor | None,           # q_plus_candidates, shape [N_active, K] or None
]:
    """Walk every completion-mask=True position across all samples and build the
    per-token V/V_target (and optionally Q+) tensors via naive forward passes.

    The "observation at active position k" is the prefix
    `prompt_ids + completion_ids[:j]` where j is the local position within the
    sample's completion (the position BEFORE the sampled token at j). This
    matches Phase 2's GAE convention `V(o_k)` = V at the state just before
    sampling the k-th action.

    Q+ at position k uses the SAME prefix, with the sampled / candidate token
    appended via `backbone.forward_q_plus(observation_ids, action_id)`. The
    candidates come from `samples[i].completion_top_k_token_ids`, which is in
    Phase 5's compact layout: outer length matches `sum(completion_mask)` for
    that sample, so the per-sample active-position counter indexes it directly.
    """
    v_list: list[float] = []
    v_target_list: list[float] = []
    q_plus_sampled_list: list[float] = []
    q_plus_candidates_list: list[list[float]] = []

    backbone.eval()
    device = next(backbone.parameters()).device

    with torch.no_grad():
        for sample in samples:
            local_active = 0  # cursor into sample.completion_top_k_token_ids
            for j, mask_bit in enumerate(sample.completion_mask):
                if not mask_bit:
                    continue
                # prefix = prompt_ids + completion_ids[:j], i.e. the state BEFORE token j.
                prefix_ids = list(sample.prompt_ids) + list(sample.completion_ids[:j])
                if not prefix_ids:
                    raise ValueError(
                        "Empty prefix at active position; samples with empty "
                        "prompt and j=0 are not supported."
                    )
                input_ids_tensor = torch.tensor(
                    [prefix_ids], dtype=torch.long, device=device
                )

                v_list.append(float(backbone.forward_v(input_ids_tensor).item()))
                v_target_list.append(
                    float(backbone.forward_v_target(input_ids_tensor).item())
                )

                if need_q_plus:
                    sampled_token = int(sample.completion_ids[j])
                    q_plus_sampled_list.append(
                        float(
                            backbone.forward_q_plus(
                                input_ids_tensor, sampled_token
                            ).item()
                        )
                    )
                    if sample.completion_top_k_token_ids is None:
                        raise ValueError(
                            "ARM compute requires sample.completion_top_k_token_ids "
                            "to be populated (Phase 5 toggle 'return_top_k_token_ids' on)."
                        )
                    candidates = sample.completion_top_k_token_ids[local_active]
                    qp_candidates_row: list[float] = []
                    for cand in candidates:
                        qp_candidates_row.append(
                            float(
                                backbone.forward_q_plus(
                                    input_ids_tensor, int(cand)
                                ).item()
                            )
                        )
                    q_plus_candidates_list.append(qp_candidates_row)

                local_active += 1

    v_all = torch.tensor(v_list, dtype=torch.float32)
    v_target_all = torch.tensor(v_target_list, dtype=torch.float32)
    if need_q_plus:
        q_plus_sampled_all = torch.tensor(q_plus_sampled_list, dtype=torch.float32)
        q_plus_candidates = torch.tensor(q_plus_candidates_list, dtype=torch.float32)
        return v_all, v_target_all, q_plus_sampled_all, q_plus_candidates
    return v_all, v_target_all, None, None


def _build_paired_outputs(
    samples: list[TrainingSample],
    advantage_out: PerTokenAdvantageOutputs,
) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
    """Splay the per-sample advantage outputs into paired
    (TrainingSample with advantages populated, AdvantageTrainingSample) tuples.

    The advantage function returns per-sample lists already at completion_ids
    granularity (Phase 2/3 splay back from the active-only GAE recursion).
    Here we just construct the paired msgspec.Struct outputs.
    """
    paired: list[tuple[TrainingSample, AdvantageTrainingSample]] = []
    for i, sample in enumerate(samples):
        advantages = advantage_out.advantages[i]
        v_targets = advantage_out.v_targets[i]
        q_plus_targets: list[float] | None = (
            advantage_out.q_plus_targets[i]
            if advantage_out.q_plus_targets is not None
            else None
        )

        # Updated TrainingSample with advantages now populated. Use msgspec.replace
        # so all other fields (top-K, routed_experts, etc.) carry over verbatim.
        import msgspec.structs

        new_llm_sample = msgspec.structs.replace(sample, advantages=advantages)

        adv_sample = AdvantageTrainingSample(
            prompt_ids=list(sample.prompt_ids),
            prompt_mask=list(sample.prompt_mask),
            completion_ids=list(sample.completion_ids),
            completion_mask=list(sample.completion_mask),
            v_targets=v_targets,
            q_plus_targets=q_plus_targets,
        )

        paired.append((new_llm_sample, adv_sample))
    return paired
