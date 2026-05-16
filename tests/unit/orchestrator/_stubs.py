"""Test fakes colocated with orchestrator-side tests.

Per stubbing-strategy section 4: shared fakes used across multiple test files
live in `_stubs.py` modules colocated with their consumers. Fakes for use
in a single test file stay inline.
"""

from __future__ import annotations

from typing import Any, Literal

from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample


def make_fake_tokens_dict(
    *,
    completion_ids: list[int],
    prompt_ids: list[int] | None = None,
    completion_mask: list[bool] | None = None,
    completion_logprobs: list[float] | None = None,
    completion_top_k_token_ids: list[list[int]] | None = None,
    routed_experts: list[list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Build a `tokens` dict shaped like what verifiers' parse_response_tokens
    produces, suitable for calls to make_sample / extend_sample in tests.
    """
    if prompt_ids is None:
        prompt_ids = [0]
    if completion_mask is None:
        completion_mask = [True] * len(completion_ids)
    if completion_logprobs is None:
        completion_logprobs = [0.0] * len(completion_ids)

    tokens: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "prompt_mask": [False] * len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_logprobs": completion_logprobs,
        "routed_experts": routed_experts,
    }
    if completion_top_k_token_ids is not None:
        tokens["completion_top_k_token_ids"] = completion_top_k_token_ids
    return tokens


class FakeInferenceServer:
    """Per stubbing-strategy section 4: deterministic stand-in for the vLLM
    inference server in orchestrator-side tests.
    """

    def __init__(self, top_k_action_set_size: int = 32):
        self.top_k_action_set_size = top_k_action_set_size

    def synthesize_top_k_for_completion(
        self, completion_ids: list[int]
    ) -> list[list[int]]:
        K = self.top_k_action_set_size
        return [[sampled + j for j in range(K)] for sampled in completion_ids]

    def make_tokens_dict(
        self,
        *,
        completion_ids: list[int],
        prompt_ids: list[int] | None = None,
        completion_mask: list[bool] | None = None,
        with_top_k: bool = True,
    ) -> dict[str, Any]:
        completion_top_k_token_ids = (
            self.synthesize_top_k_for_completion(completion_ids)
            if with_top_k
            else None
        )
        return make_fake_tokens_dict(
            completion_ids=completion_ids,
            prompt_ids=prompt_ids,
            completion_mask=completion_mask,
            completion_top_k_token_ids=completion_top_k_token_ids,
        )


class FakeAdvantageServer:
    """In-process drop-in for the Advantage Server (no real HTTP).

    Implements `AdvantageServerClientProtocol`'s interface so the orchestrator
    can be parameterized over either the real httpx-backed `AdvantageServerClient`
    or this fake.

    The fake's behavior at the iteration-0 regime: every paired output has
    zero `advantages`, zero `v_targets`, and zero `q_plus_targets` (when
    algorithm is "arm"). This matches what a real Advantage Server backed by
    a freshly-zero-init'd ValueNetworkBackbone would produce on tiny
    synthetic trajectories. The exact zero is what Phase 3's cold-start
    branch outputs and what Phase 2's GAE produces with V_all == 0.

    Tests that need non-zero outputs can pass an `output_factory` that
    constructs paired outputs from the input samples; this lets a test
    pin specific values (e.g., for the sync invariant test that needs to
    verify the prompt/completion fields propagate verbatim).

    Weight-step state: mirrors the real AdvServer's `app.state.weight_step`.
    Starts at 0; tests can call `advance_weight_step()` to simulate the
    AdvTrainer broadcasting a new state_dict. `wait_for_weight_step`
    short-circuits when satisfied or polls until the condition is met (or
    timeout). For tests that need to verify the orchestrator actually waits,
    use `weight_step_event` to manually unblock the wait at a chosen point
    in the test.
    """

    def __init__(
        self,
        *,
        output_factory: (
            Any | None
        ) = None,  # callable: (samples, ep_reward, is_terminal, algorithm) -> list[(TrainingSample, AdvantageTrainingSample)]
        initial_weight_step: int = 0,
    ):
        self._output_factory = output_factory
        self.calls: list[dict[str, Any]] = []
        # AdvTrainer broadcast counter. Tests advance this manually to
        # simulate AdvTrainer broadcasts; the orchestrator's
        # wait_for_weight_step polls this value.
        self.weight_step: int = initial_weight_step
        self.wait_calls: list[dict[str, Any]] = []

    def advance_weight_step(self, n: int = 1) -> int:
        """Simulate `n` AdvTrainer broadcasts. Returns the new weight_step."""
        self.weight_step += n
        return self.weight_step

    async def compute_advantages_and_targets(
        self,
        samples: list[TrainingSample],
        episodic_reward: float,
        is_terminal: bool,
        algorithm: Literal["ppo", "arm"],
    ) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
        """Return per-input-sample paired outputs. Records the call for assertions.

        Also records the weight_step at the moment of the call -- tests can
        assert "compute was only called when the AdvServer had received the
        expected number of broadcasts."
        """
        self.calls.append(
            {
                "samples": samples,
                "episodic_reward": episodic_reward,
                "is_terminal": is_terminal,
                "algorithm": algorithm,
                "weight_step_at_call": self.weight_step,
            }
        )
        if self._output_factory is not None:
            return self._output_factory(samples, episodic_reward, is_terminal, algorithm)
        # Default: zeros at every position; mirrors the iteration-0 regime
        # produced by ValueNetworkBackbone with zero-init adapters/heads.
        return [_zero_paired_for_sample(s, algorithm) for s in samples]

    async def aclose(self) -> None:
        pass

    async def wait_for_ready(
        self, timeout: float = 600.0, poll_interval: float = 1.0
    ) -> None:
        """Always ready; no real server to poll."""
        return None

    async def wait_for_weight_step(
        self,
        min_step: int,
        timeout: float = 3600.0,
        poll_interval: float = 1.0,
    ) -> int:
        """Block until self.weight_step >= min_step (or raise on timeout).

        Records the call for assertions. Polls with `asyncio.sleep` so other
        coroutines (e.g., a simulated AdvTrainer that calls
        `advance_weight_step` after some delay) can run between polls.

        For step 0 the condition is trivially satisfied (weight_step starts
        at 0 >= 0); we still record the call so tests can assert the gate
        was hit. This matches the real client's behavior of short-circuiting
        when min_step <= 0.
        """
        import asyncio

        self.wait_calls.append(
            {
                "min_step": min_step,
                "weight_step_at_entry": self.weight_step,
            }
        )
        if min_step <= 0:
            return self.weight_step

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while self.weight_step < min_step:
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"FakeAdvantageServer.wait_for_weight_step: weight_step "
                    f"{self.weight_step} < {min_step} after {timeout}s"
                )
            await asyncio.sleep(poll_interval)
        return self.weight_step


def _zero_paired_for_sample(
    sample: TrainingSample,
    algorithm: Literal["ppo", "arm"],
) -> tuple[TrainingSample, AdvantageTrainingSample]:
    """Construct a (TrainingSample-with-zero-advantages, AdvantageTrainingSample-with-zeros) pair."""
    import msgspec.structs

    n = len(sample.completion_ids)
    zero_per_token = [0.0] * n
    new_llm_sample = msgspec.structs.replace(sample, advantages=list(zero_per_token))
    adv_sample = AdvantageTrainingSample(
        prompt_ids=list(sample.prompt_ids),
        prompt_mask=list(sample.prompt_mask),
        completion_ids=list(sample.completion_ids),
        completion_mask=list(sample.completion_mask),
        v_targets=list(zero_per_token),
        q_plus_targets=list(zero_per_token) if algorithm == "arm" else None,
    )
    return new_llm_sample, adv_sample
