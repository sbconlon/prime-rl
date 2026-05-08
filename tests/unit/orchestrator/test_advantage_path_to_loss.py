"""Phase 8 -- end-to-end advantage path: Advantage Server -> prepare_sample -> default_loss_fn.

These integration-shaped unit tests walk the full per-token-advantages path
in a single test:

    LLMTrainingSample (Phase 1 contract)
        -> Advantage Server (faked)
            -> prepare_sample (Phase 1 modified)
                -> MicroBatch.advantages
                    -> LossInputs
                        -> default_loss_fn (Phase 8 verified)

Earlier phases test each link in isolation; Phase 8\'s integration test
catches any latent shape/type mismatch that could have escaped phase-local
coverage. CPU-only.

See plan/phases/phase-08-llm-loss-verification.md.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.batch import prepare_sample
from prime_rl.trainer.rl.loss import LossInputs, default_loss_fn
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample
from tests.unit.orchestrator._stubs import FakeAdvantageServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_training_sample(
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    advantages: list[float] | None = None,
) -> TrainingSample:
    n_completion = len(completion_ids)
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * n_completion,
        completion_logprobs=[0.0] * n_completion,
        completion_temperatures=[1.0] * n_completion,
        advantages=advantages,
    )


def _loss_inputs_from_micro_batch(
    *,
    advantages: list[float],
    loss_mask: list[bool],
    trainer_offset: float = -0.10,
    inference_offset: float = -0.15,
) -> LossInputs:
    """Build a LossInputs whose ``advantages`` and ``loss_mask`` come from a
    MicroBatch. trainer/inference logprobs are constants (the Phase 8 tests
    verify the loss handles per-token advantages, not log-probabilities)."""
    seq_len = len(advantages)
    return LossInputs(
        trainer_logprobs=torch.full((seq_len,), trainer_offset, dtype=torch.float32),
        inference_logprobs=torch.full((seq_len,), inference_offset, dtype=torch.float32),
        teacher_logprobs=None,
        advantages=torch.tensor(advantages, dtype=torch.float32),
        loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
    )


def _output_factory_with_advantages(
    advantages_per_sample: list[list[float]],
):
    """Return a FakeAdvantageServer output_factory that injects fixed
    per-token advantages into the LLMTrainingSample side of each paired
    output. ``v_targets`` / ``q_plus_targets`` on the AdvantageTrainingSample
    side are zero-filled (they\'re not exercised by Phase 8\'s loss path)."""

    def factory(
        samples: list[TrainingSample],
        episodic_reward: float,
        is_terminal: bool,
        algorithm: Literal["ppo", "arm"],
    ) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
        assert len(samples) == len(advantages_per_sample), (
            f"output_factory: got {len(samples)} samples but "
            f"{len(advantages_per_sample)} advantage lists"
        )
        out: list[tuple[TrainingSample, AdvantageTrainingSample]] = []
        for sample, advs in zip(samples, advantages_per_sample):
            assert len(advs) == len(sample.completion_ids), (
                f"advantages length {len(advs)} != completion_ids length "
                f"{len(sample.completion_ids)}"
            )
            llm = TrainingSample(
                prompt_ids=sample.prompt_ids,
                prompt_mask=sample.prompt_mask,
                completion_ids=sample.completion_ids,
                completion_mask=sample.completion_mask,
                completion_logprobs=sample.completion_logprobs,
                completion_temperatures=sample.completion_temperatures,
                advantages=list(advs),
            )
            adv = AdvantageTrainingSample(
                prompt_ids=sample.prompt_ids,
                prompt_mask=sample.prompt_mask,
                completion_ids=sample.completion_ids,
                completion_mask=sample.completion_mask,
                v_targets=[0.0] * len(sample.completion_ids),
                q_plus_targets=(
                    [0.0] * len(sample.completion_ids) if algorithm == "arm" else None
                ),
            )
            out.append((llm, adv))
        return out

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_advantage_server_output_consumed_by_loss():
    """Full path: TrainingSample(advantages=None) -> FakeAdvantageServer
    populates per-token advantages -> prepare_sample produces the
    prompt-zero-padded MicroBatch.advantages -> default_loss_fn consumes it
    and returns finite loss. Catches any shape/type mismatch in the chain."""
    sample = _make_training_sample(
        prompt_ids=[1, 2],
        completion_ids=[10, 20, 30],
        advantages=None,
    )

    fake = FakeAdvantageServer(
        output_factory=_output_factory_with_advantages([[0.5, -0.3, 0.8]])
    )
    paired = asyncio.run(
        fake.compute_advantages_and_targets(
            samples=[sample],
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="ppo",
        )
    )
    llm_sample, _ = paired[0]
    assert llm_sample.advantages == [0.5, -0.3, 0.8]

    micro_batch = prepare_sample(llm_sample, seq_len=64)
    # Phase 1 invariant: per-token advantages with zeros at prompt positions.
    assert micro_batch.advantages == [0.0, 0.0, 0.5, -0.3, 0.8]
    assert len(micro_batch.advantages) == len(micro_batch.input_ids)

    inputs = _loss_inputs_from_micro_batch(
        advantages=micro_batch.advantages,
        loss_mask=micro_batch.loss_mask,
    )
    out = default_loss_fn(
        inputs, DefaultLossConfig(ipo_mask_high=10.0, ipo_mask_low=10.0)
    )

    assert torch.isfinite(out.loss).all(), f"loss not finite: {out.loss}"
    for name, value in out.metrics.items():
        assert torch.isfinite(value).all(), f"metric {name} not finite: {value}"


def test_grpo_path_and_arm_zero_path_produce_equivalent_loss():
    """The loss must be algorithm-agnostic: when GRPO\'s scalar broadcast
    and ARM\'s per-token output happen to coincide (e.g., uniform 0.5 across
    completion positions), the resulting loss must be bit-equal. Grounds the
    "loss is opaque to algorithm" claim in a mechanically-verifiable check.

    Both paths produce TrainingSample.advantages = [0.5, 0.5, 0.5]; the only
    difference would be in how upstream code arrived at that list. prepare_sample
    + default_loss_fn must not care.
    """
    grpo_sample = _make_training_sample(
        prompt_ids=[1, 2],
        completion_ids=[10, 20, 30],
        advantages=[0.5, 0.5, 0.5],  # GRPO scalar 0.5 broadcast across completion
    )

    # ARM cold-start path: produced by FakeAdvantageServer with an output_factory
    # that returns the same uniform values. Different upstream provenance,
    # identical TrainingSample state.
    base_sample = _make_training_sample(
        prompt_ids=[1, 2],
        completion_ids=[10, 20, 30],
        advantages=None,
    )
    fake = FakeAdvantageServer(
        output_factory=_output_factory_with_advantages([[0.5, 0.5, 0.5]])
    )
    arm_paired = asyncio.run(
        fake.compute_advantages_and_targets(
            samples=[base_sample],
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="arm",
        )
    )
    arm_sample, _ = arm_paired[0]

    # Phase 1 invariant: state matches.
    assert grpo_sample.advantages == arm_sample.advantages

    grpo_mb = prepare_sample(grpo_sample, seq_len=64)
    arm_mb = prepare_sample(arm_sample, seq_len=64)
    assert grpo_mb.advantages == arm_mb.advantages

    cfg = DefaultLossConfig(ipo_mask_high=10.0, ipo_mask_low=10.0)
    grpo_loss = default_loss_fn(
        _loss_inputs_from_micro_batch(
            advantages=grpo_mb.advantages, loss_mask=grpo_mb.loss_mask
        ),
        cfg,
    ).loss
    arm_loss = default_loss_fn(
        _loss_inputs_from_micro_batch(
            advantages=arm_mb.advantages, loss_mask=arm_mb.loss_mask
        ),
        cfg,
    ).loss

    assert torch.allclose(grpo_loss, arm_loss, atol=0, rtol=0), (
        f"GRPO and ARM-cold-start loss must be bit-equal when advantages "
        f"coincide; got grpo={grpo_loss.item()}, arm={arm_loss.item()}"
    )
