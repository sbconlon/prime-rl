"""Advantage Server compute logic.

Given a `list[TrainingSample]` from a single rollout (post-`interleave_rollout`,
each with `advantages=None`) plus rollout metadata (`episodic_reward`,
`is_terminal`), produce paired `(TrainingSample, AdvantageTrainingSample)`
outputs.

Phase 6.5 implementation: optimized forward path. Three prefix forwards
(V, V_target, Q+) populate per-adapter caches and produce all-positions
values. ARM additionally evaluates K candidates per position via batched
forward with the cloned-cropped-tiled Q+ cache. The advantage functions
themselves (Phase 2/3) are unchanged -- they consume the same per-token
V/V_target/Q+ tensors as before.

Phase 6's naive per-position forwards on `ValueNetworkBackbone` are
retained as the correctness oracle for the optimized path; see
`test_optimized_compute_*_matches_naive_compute_*` in test_compute.py.

Sample-type note: Phase 6's wire type is `TrainingSample` (the canonical
sample type from Phase 1); the master plan's `LLMTrainingSample` rename is
a deferred edit pass.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

_LOGGER = logging.getLogger("prime_rl.advantage_server")

from prime_rl.orchestrator.per_token_advantage import (
    ArmAdvantageInputs,
    PerTokenAdvantageOutputs,
    PpoAdvantageInputs,
    arm_regret_matching_advantage_fn,
    ppo_gae_advantage_fn,
)
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.advantage_server._prof import prof
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
    """PPO GAE per-token advantages + v_targets for one rollout, optimized path.

    Two adapter-prefix forwards (V, V_target) over the per-sample sequence
    produce all-position V / V_target values. The Phase 2 advantage function
    consumes those tensors and returns advantages and v_targets. The naive
    per-position-forward path is preserved on `ValueNetworkBackbone` as a
    correctness oracle for tests.
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
    """ARM regret-matching per-token advantages + v_targets + q_plus_targets,
    optimized path.

    Three adapter-prefix forwards (V, V_target, Q+) plus per-position batched
    K-candidate evaluation against a cloned-cropped-tiled Q+ cache. The
    Phase 3 advantage function consumes the resulting tensors.
    """
    n_samples = len(samples)
    total_completion = sum(len(s.completion_ids) for s in samples)
    total_active = sum(sum(s.completion_mask) for s in samples)
    with prof(
        "compute.arm.TOTAL",
        n_samples=n_samples,
        total_completion=total_completion,
        total_active=total_active,
    ):
        with prof("compute.arm.build_value_tensors", sync_cuda=True):
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
        with prof("compute.arm.regret_matching"):
            out = arm_regret_matching_advantage_fn(inputs)
        # Phase 10 diagnostic: per-request advantage + target magnitudes.
        # Catches the case where regret-matching produces ~0 (cold start,
        # or normalization that eats signal) before that signal reaches the
        # LLM trainer. Look for this line in advantage_server.log.
        _log_advantage_magnitudes(out)
        with prof("compute.arm.build_paired_outputs"):
            return _build_paired_outputs(samples, out)


def _log_advantage_magnitudes(out: PerTokenAdvantageOutputs) -> None:
    """Emit per-request advantage / v_target / q_plus_target magnitudes."""
    import math
    adv_flat = [a for sample in out.advantages for a in sample]
    v_flat = [v for sample in out.v_targets for v in sample]
    if not adv_flat:
        _LOGGER.info("ARM_DIAG advantages=empty")
        return
    adv_abs = [abs(a) for a in adv_flat]
    v_abs = [abs(v) for v in v_flat]
    msg = (
        f"ARM_DIAG n_tokens={len(adv_flat)} "
        f"adv_abs_mean={sum(adv_abs)/len(adv_abs):.4e} "
        f"adv_abs_max={max(adv_abs):.4e} "
        f"v_target_abs_mean={sum(v_abs)/len(v_abs):.4e}"
    )
    if out.q_plus_targets is not None:
        q_flat = [q for sample in out.q_plus_targets for q in sample]
        q_abs = [abs(q) for q in q_flat]
        msg += f" q_plus_target_abs_mean={sum(q_abs)/len(q_abs):.4e}"
    _LOGGER.info(msg)


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
# Internal helpers (Phase 6.5 optimized path)
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
    """Build per-token V/V_target (and optionally Q+) tensors using the
    optimized forward methods on `ValueNetworkBackbone`.

    Strategy:
      For each sample i, run three (or two for PPO) prefix forwards
      separately over `prompt_ids + completion_ids`. Collect per-token V /
      V_target / Q+_sampled values. For ARM, also evaluate K candidates per
      mask=True position using the cloned+cropped+tiled Q+ cache.

    Per-sample forwards rather than one joint forward across samples: each
    sample has its own prompt_ids that may differ (different turn boundaries
    in fragmented rollouts). The Phase 2/3 advantage functions concatenate
    the resulting per-token tensors before the GAE / regret-matching
    recursion, so the joint trajectory view is preserved at the math layer.

    Output shapes are length `N_active = sum(sum(s.completion_mask) for s in samples)`,
    matching what the Phase 2/3 advantage functions expect. Mask=False
    positions inside `completion_ids` (e.g. injected next-turn prompts in
    fragmented rollouts) are skipped here.
    """
    backbone.eval()
    device = next(backbone.parameters()).device

    v_chunks: list[torch.Tensor] = []
    v_target_chunks: list[torch.Tensor] = []
    q_plus_sampled_chunks: list[torch.Tensor] = []
    q_plus_candidates_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for sample in samples:
            prompt_len = len(sample.prompt_ids)
            completion_len = len(sample.completion_ids)
            full_input_ids = torch.tensor(
                [list(sample.prompt_ids) + list(sample.completion_ids)],
                dtype=torch.long,
                device=device,
            )

            # ----- V all-positions -----
            with prof(
                "build_value_tensors.forward_v_all",
                sync_cuda=True,
                prompt_len=prompt_len,
                completion_len=completion_len,
            ):
                v_all_seq, _v_cache = backbone.forward_v_all_positions(full_input_ids)
            # v_all_seq[0, k] is V at observation o_k = first k tokens of input.
            # We need V(o_k) for k = (prompt_len-1, ..., prompt_len + completion_len - 2)
            # because position 0 of completion is "before completion[0] is sampled,
            # the model has just observed prompt's last token". That's
            # v_all_seq[0, prompt_len - 1].
            # In general V at completion position j (0-indexed within completion)
            # is read at full-input position `prompt_len - 1 + j`.
            # See compute.py PHASE 6 for the same convention.
            v_per_completion = v_all_seq[0, prompt_len - 1 : prompt_len - 1 + completion_len]

            # ----- V_target all-positions -----
            with prof(
                "build_value_tensors.forward_v_target_all",
                sync_cuda=True,
                prompt_len=prompt_len,
                completion_len=completion_len,
            ):
                v_target_all_seq, _vt_cache = backbone.forward_v_target_all_positions(full_input_ids)
            v_target_per_completion = v_target_all_seq[
                0, prompt_len - 1 : prompt_len - 1 + completion_len
            ]

            if need_q_plus:
                # ----- Q+ all-positions (sampled) + K-candidate per active position -----
                with prof(
                    "build_value_tensors.forward_q_plus_sampled_all",
                    sync_cuda=True,
                    prompt_len=prompt_len,
                    completion_len=completion_len,
                ):
                    q_plus_all_seq, q_plus_cache = backbone.forward_q_plus_sampled_all_positions(
                        full_input_ids
                    )
                # Q+ at the sampled action a_k = completion_ids[k] is read at the
                # full-input position right AFTER token a_k, i.e. position
                # `prompt_len + k`. (One position later than V, because Q+
                # represents "after observing the action.")
                q_plus_sampled_per_completion = q_plus_all_seq[
                    0, prompt_len : prompt_len + completion_len
                ]

                # K candidates per active (mask=True) position. Plan §4.5
                # batched path: all (j, c) pairs run through a single
                # forward_q_plus_candidates_batched call sharing the same
                # physical q_plus_cache via cache_batch_idx=zeros(N_active * K).
                # Replaces the per-position loop that previously dominated
                # ~95% of request latency in the canary.
                K = (
                    len(sample.completion_top_k_token_ids[0])
                    if sample.completion_top_k_token_ids
                    else 0
                )
                if K == 0:
                    raise ValueError(
                        "ARM compute requires sample.completion_top_k_token_ids "
                        "to be populated (Phase 5 toggle 'return_top_k_token_ids' on)."
                    )

                # Collect active positions in compact layout (Phase 5 Option C).
                active_local_js = [
                    j for j, m in enumerate(sample.completion_mask) if m
                ]
                n_active = len(active_local_js)

                if n_active > 0:
                    # Build candidate tensors. completion_top_k_token_ids is
                    # in compact layout: index i corresponds to the i-th
                    # mask=True position.
                    candidate_token_ids = torch.tensor(
                        [
                            list(sample.completion_top_k_token_ids[i])
                            for i in range(n_active)
                        ],
                        dtype=torch.long,
                        device=device,
                    )  # [N_active, K]
                    candidate_positions = torch.tensor(
                        [prompt_len + j for j in active_local_js],
                        dtype=torch.int32,
                        device=device,
                    )  # [N_active]

                    with prof(
                        "build_value_tensors.candidates_batched",
                        sync_cuda=True,
                        n_active=n_active,
                        K=K,
                    ):
                        q_for_all = backbone.forward_q_plus_candidates_batched(
                            prefix_cache=q_plus_cache,
                            candidate_token_ids=candidate_token_ids,
                            candidate_positions=candidate_positions,
                        )  # [N_active, K]

                    # Q+ at the sampled action for each active position:
                    # gather from the per-completion sampled tensor.
                    active_js_tensor = torch.tensor(
                        active_local_js, dtype=torch.long, device=device
                    )
                    per_position_sampled = q_plus_sampled_per_completion[active_js_tensor]
                    q_plus_sampled_chunks.append(per_position_sampled)
                    q_plus_candidates_chunks.append(q_for_all)

            # Splay V / V_target into mask=True positions only (compact form).
            for j, mask_bit in enumerate(sample.completion_mask):
                if not mask_bit:
                    continue
                v_chunks.append(v_per_completion[j].detach().cpu().float())
                v_target_chunks.append(v_target_per_completion[j].detach().cpu().float())

    if not v_chunks:
        # No mask=True positions across all samples -- return empty tensors.
        empty = torch.zeros(0, dtype=torch.float32)
        if need_q_plus:
            return empty, empty, empty, torch.zeros(0, 0, dtype=torch.float32)
        return empty, empty, None, None

    v_all = torch.stack(v_chunks).to(dtype=torch.float32)
    v_target_all = torch.stack(v_target_chunks).to(dtype=torch.float32)
    if need_q_plus:
        q_plus_sampled_all = (
            torch.cat(q_plus_sampled_chunks).detach().cpu().float()
        )
        q_plus_candidates = (
            torch.cat(q_plus_candidates_chunks).detach().cpu().float()
        )
        return v_all, v_target_all, q_plus_sampled_all, q_plus_candidates
    return v_all, v_target_all, None, None


def _build_paired_outputs(
    samples: list[TrainingSample],
    advantage_out: PerTokenAdvantageOutputs,
) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
    """Splay the per-sample advantage outputs into paired
    (TrainingSample with advantages populated, AdvantageTrainingSample) tuples.
    """
    import msgspec.structs

    paired: list[tuple[TrainingSample, AdvantageTrainingSample]] = []
    for i, sample in enumerate(samples):
        advantages = advantage_out.advantages[i]
        v_targets = advantage_out.v_targets[i]
        q_plus_targets: list[float] | None = (
            advantage_out.q_plus_targets[i]
            if advantage_out.q_plus_targets is not None
            else None
        )

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
