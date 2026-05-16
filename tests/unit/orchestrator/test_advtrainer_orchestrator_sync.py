"""AdvTrainer<->orchestrator lockstep barrier tests.

The orchestrator's per-step body inserts an
`await advantage_server_client.wait_for_weight_step(min_step=progress.step)`
gate between rollout preprocessing and the per-rollout AdvServer compute
calls. This file pins the invariant the gate enforces:

  At any moment the orchestrator is issuing step-N AdvServer compute
  calls, the AdvServer has applied the AdvTrainer's broadcast resulting
  from training on batch N-1 (i.e., AdvServer.weight_step >= N).

We test this in three layers:

  1. The barrier invariant in isolation: confirm that `compute_advantages_and_targets`
     never sees a weight_step < min_step when paired with `wait_for_weight_step`.
  2. The lockstep simulation: drive a simulated multi-step orchestrator loop
     against a FakeAdvantageServer + a simulated AdvTrainer that only advances
     weight_step after a delay. Assert weight_step on every compute call
     matches the step counter.
  3. The cross-step staleness invariant: assert no advantage sample from
     an earlier orchestrator step bleeds into a later step's batch
     (the per-step `advantage_examples` local-list pattern).

These are CPU unit tests; no real subprocess, no GPU, no real model. The
mechanism under test is a sleep-gated async polling pattern, so the tests
use short timeouts and `asyncio.gather` to interleave coroutines.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample
from tests.unit.orchestrator._stubs import FakeAdvantageServer


def _make_sample(completion_len: int, *, prompt_len: int = 2) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(1, prompt_len + 1)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
    )


# ---------------------------------------------------------------------------
# 1. Barrier invariant: compute_advantages_and_targets never sees a stale
#    weight_step when preceded by wait_for_weight_step.
# ---------------------------------------------------------------------------


def test_compute_only_runs_after_wait_satisfies_min_step():
    """If the orchestrator awaits wait_for_weight_step(min_step=N) before any
    compute call, every recorded compute call has weight_step_at_call >= N.

    This is the precise invariant we want from the orchestrator's gate:
    the gate must enforce that the AdvServer has received the expected
    number of broadcasts BEFORE any compute call for the next step fires.
    """
    fake = FakeAdvantageServer()  # weight_step=0
    samples = [_make_sample(completion_len=2)]

    async def step_with_gate(step_num: int):
        await fake.wait_for_weight_step(min_step=step_num)
        await fake.compute_advantages_and_targets(
            samples=samples,
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="ppo",
        )

    async def simulated_advtrainer():
        # Stagger the broadcasts: each cycle takes ~30ms wall-clock.
        # The orchestrator's "step k" gate waits for broadcast k to land.
        for _ in range(5):
            await asyncio.sleep(0.03)
            fake.advance_weight_step(1)

    async def run():
        trainer = asyncio.create_task(simulated_advtrainer())
        # Drive 5 orchestrator steps (steps 0..4). step 0's gate is
        # min_step=0 which short-circuits; subsequent steps block until
        # the AdvTrainer advances.
        for step in range(5):
            await step_with_gate(step)
        await trainer

    asyncio.run(run())

    # For each compute call (one per step), weight_step at call time
    # must equal the orchestrator's step number (>= the min_step gate).
    assert len(fake.calls) == 5
    for i, call in enumerate(fake.calls):
        assert call["weight_step_at_call"] >= i, (
            f"step {i}: compute fired with weight_step="
            f"{call['weight_step_at_call']}, expected >= {i}"
        )


def test_compute_blocked_until_advtrainer_broadcasts():
    """A simulated 'slow AdvTrainer' that broadcasts only after a delay
    pins the gate's blocking behavior: compute call does not fire while
    weight_step is still stale.
    """
    fake = FakeAdvantageServer()  # weight_step=0

    timeline: list[tuple[float, str]] = []
    t0 = None

    async def gate_and_compute():
        nonlocal t0
        loop = asyncio.get_event_loop()
        if t0 is None:
            t0 = loop.time()
        timeline.append((loop.time() - t0, "wait_start"))
        await fake.wait_for_weight_step(min_step=1, poll_interval=0.005)
        timeline.append((loop.time() - t0, "wait_done"))
        await fake.compute_advantages_and_targets(
            samples=[_make_sample(completion_len=1)],
            episodic_reward=1.0,
            is_terminal=True,
            algorithm="ppo",
        )
        timeline.append((loop.time() - t0, "compute_done"))

    async def advtrainer_after_delay():
        loop = asyncio.get_event_loop()
        await asyncio.sleep(0.1)
        nonlocal_t0 = loop.time() - (t0 if t0 is not None else loop.time())
        timeline.append((nonlocal_t0, "advtrainer_broadcast"))
        fake.advance_weight_step(1)

    async def run():
        await asyncio.gather(gate_and_compute(), advtrainer_after_delay())

    asyncio.run(run())

    # Timeline ordering: wait_start -> advtrainer_broadcast -> wait_done
    # -> compute_done. The compute call MUST happen after the broadcast.
    events = [name for _, name in timeline]
    assert events.index("wait_start") < events.index("advtrainer_broadcast"), (
        f"Expected wait to start before broadcast. Timeline: {timeline}"
    )
    assert events.index("advtrainer_broadcast") < events.index("wait_done"), (
        f"Expected broadcast before wait_done (gate must block until "
        f"advance). Timeline: {timeline}"
    )
    assert events.index("wait_done") < events.index("compute_done"), (
        f"Expected wait_done before compute_done. Timeline: {timeline}"
    )

    # Compute saw the post-broadcast weight_step.
    assert fake.calls[0]["weight_step_at_call"] == 1


# ---------------------------------------------------------------------------
# 2. Lockstep simulation across many steps
# ---------------------------------------------------------------------------


def test_lockstep_simulation_across_many_steps():
    """Drive 8 orchestrator steps + a simulated AdvTrainer that broadcasts
    once per step. Assert every compute call observes weight_step matching
    the orchestrator's current step.

    Inverts the trainer/orchestrator speed relationship across runs to
    cover both regimes:
      a) AdvTrainer slower: orchestrator blocks; compute is delayed.
      b) AdvTrainer faster: orchestrator skips the wait; compute fires
         immediately with a still-correct weight_step.
    """

    async def run_one(trainer_delay: float, orch_delay: float):
        fake = FakeAdvantageServer()
        n_steps = 8

        async def orch():
            for step in range(n_steps):
                # Orchestrator's "step setup" takes orch_delay (rollout gen,
                # preprocessing).
                await asyncio.sleep(orch_delay)
                await fake.wait_for_weight_step(min_step=step, poll_interval=0.005)
                await fake.compute_advantages_and_targets(
                    samples=[_make_sample(completion_len=1)],
                    episodic_reward=1.0,
                    is_terminal=True,
                    algorithm="ppo",
                )

        async def trainer():
            # Step 0's broadcast lands at the END of step 0 (not before),
            # so the AdvTrainer's first broadcast comes after the
            # orchestrator's first compute. Subsequent broadcasts advance
            # weight_step by 1 each.
            await asyncio.sleep(orch_delay + 0.005)  # let orch's step 0 enter compute
            for _ in range(n_steps - 1):
                await asyncio.sleep(trainer_delay)
                fake.advance_weight_step(1)

        await asyncio.gather(orch(), trainer())
        return fake

    # (a) AdvTrainer slower than orchestrator.
    fake_a = asyncio.run(run_one(trainer_delay=0.05, orch_delay=0.01))
    assert len(fake_a.calls) == 8
    for i, call in enumerate(fake_a.calls):
        assert call["weight_step_at_call"] == i, (
            f"slow-trainer regime, step {i}: weight_step="
            f"{call['weight_step_at_call']}, expected exactly {i} "
            f"(lockstep). Timeline of calls: "
            f"{[c['weight_step_at_call'] for c in fake_a.calls]}"
        )

    # (b) AdvTrainer faster than orchestrator. The gate still holds:
    # weight_step at call time is >= the step number, but may be greater
    # if the AdvTrainer raced ahead. Under the actual orchestrator code,
    # this can't happen because the orchestrator only sends batch N+1
    # AFTER the AdvServer has advanced past step N+1 -- the AdvTrainer
    # can't broadcast for step N+2 without orchestrator sending batch
    # N+1 first. So in production weight_step_at_call == step exactly.
    # In the simulation here we tolerate >= because we don't simulate the
    # full AdvTrainer-blocked-on-orchestrator-output coupling.
    fake_b = asyncio.run(run_one(trainer_delay=0.005, orch_delay=0.05))
    assert len(fake_b.calls) == 8
    for i, call in enumerate(fake_b.calls):
        assert call["weight_step_at_call"] >= i, (
            f"fast-trainer regime, step {i}: weight_step="
            f"{call['weight_step_at_call']}, expected >= {i}"
        )


# ---------------------------------------------------------------------------
# 3. Cross-step advantage_examples staleness invariant
# ---------------------------------------------------------------------------


def test_advantage_examples_local_per_step_no_cross_step_bleed():
    """Confirm the orchestrator's `advantage_examples = []` pattern is
    step-local: simulate the per-step accumulation and the assembly of
    the AdvantageTrainingBatch, then assert that step N's batch never
    contains samples produced during step M != N.

    This mirrors orchestrator.py's per-step body where:
      advantage_examples: list[AdvantageTrainingSample] = []  (fresh per step)
      for rollout in rollouts_for_this_step:
          paired = await advantage_server_client.compute_advantages_and_targets(...)
          advantage_examples.extend(p[1] for p in paired)
      batch = AdvantageTrainingBatch(examples=advantage_examples, step=...)
    """
    fake = FakeAdvantageServer()
    batches_sent: list[tuple[int, list[AdvantageTrainingSample]]] = []

    async def simulated_step(step_num: int, samples_for_this_step: list[TrainingSample]):
        # Mirror the orchestrator's per-step local accumulation pattern.
        advantage_examples: list[AdvantageTrainingSample] = []
        await fake.wait_for_weight_step(min_step=step_num)
        for s in samples_for_this_step:
            paired = await fake.compute_advantages_and_targets(
                samples=[s],
                episodic_reward=float(step_num),
                is_terminal=True,
                algorithm="ppo",
            )
            advantage_examples.extend(pair[1] for pair in paired)
        batches_sent.append((step_num, list(advantage_examples)))

    async def run():
        # 3 steps, each producing a uniquely identifiable batch via
        # episodic_reward proxy (samples differ per step too).
        await simulated_step(0, [
            _make_sample(completion_len=2, prompt_len=2),
            _make_sample(completion_len=3, prompt_len=2),
        ])
        fake.advance_weight_step(1)
        await simulated_step(1, [_make_sample(completion_len=5, prompt_len=3)])
        fake.advance_weight_step(1)
        await simulated_step(2, [
            _make_sample(completion_len=1, prompt_len=4),
            _make_sample(completion_len=4, prompt_len=4),
            _make_sample(completion_len=2, prompt_len=4),
        ])

    asyncio.run(run())

    # 3 batches with the expected sample counts per step.
    assert len(batches_sent) == 3
    assert len(batches_sent[0][1]) == 2
    assert len(batches_sent[1][1]) == 1
    assert len(batches_sent[2][1]) == 3

    # Identity invariant: each step's batch only contains samples whose
    # prompt_len matches the step's prompt_len convention (2/3/4).
    # This proves there's no cross-step bleed.
    expected_prompt_lens = {0: 2, 1: 3, 2: 4}
    for step_num, batch_examples in batches_sent:
        expected = expected_prompt_lens[step_num]
        for adv_sample in batch_examples:
            assert len(adv_sample.prompt_ids) == expected, (
                f"step {step_num}: found an AdvantageTrainingSample with "
                f"prompt_len={len(adv_sample.prompt_ids)}, expected {expected} "
                f"(cross-step bleed?)"
            )


# ---------------------------------------------------------------------------
# 4. Step-0 bootstrap path: gate doesn't block on the cold start
# ---------------------------------------------------------------------------


def test_step_0_does_not_block_on_initial_weight_step():
    """At orchestrator step 0, the AdvServer's weight_step=0 (initial,
    pre-broadcast). The gate's min_step=0 must short-circuit immediately
    so the zero-init backbone's outputs are used for the bootstrap
    compute. Without this, the pipeline would deadlock on step 0
    (no broadcast yet possible -- AdvTrainer needs batch 0's data
    which can't exist until orchestrator step 0 completes).
    """
    fake = FakeAdvantageServer()  # weight_step=0
    samples = [_make_sample(completion_len=1)]

    async def run():
        # Should complete in negligible time -- no polling, no advance needed.
        result = await fake.wait_for_weight_step(min_step=0, timeout=0.5)
        assert result == 0
        # And the compute call uses the initial weights.
        paired = await fake.compute_advantages_and_targets(
            samples=samples, episodic_reward=1.0, is_terminal=True, algorithm="ppo",
        )
        return paired

    paired = asyncio.run(run())
    assert len(paired) == 1
    assert fake.calls[0]["weight_step_at_call"] == 0
