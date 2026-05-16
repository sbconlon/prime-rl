"""HTTP client for the Advantage Server.

Phase 6a defines the wire-protocol request/response msgspec types and the
async client. The endpoint itself ships in Phase 6b. The client is fully
testable in 6a via `FakeAdvantageServer` (in-process drop-in).

Per-rollout payload contract (DQ13): one HTTP call per rollout, carrying all
N samples that `interleave_rollout` produced for that rollout. The server
processes them jointly so the GAE / regret-matching recursion spans the full
trajectory across fragment boundaries; the response carries paired
(TrainingSample-with-advantages, AdvantageTrainingSample) outputs in matching
order.
"""

from __future__ import annotations

from typing import Literal, Protocol

import httpx
import msgspec

from prime_rl.configs.advantage_server import AdvantageServerClientConfig
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample

# ---------------------------------------------------------------------------
# Wire-protocol types (msgspec.Struct so they share the existing serialization
# stack with TrainingBatch / TrainingSample).
# ---------------------------------------------------------------------------


class ComputeAdvantagesRequest(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """Request body for POST /compute_advantages_and_targets.

    Every sample's `advantages` field must be None on entry (the server populates
    them in the response).
    """

    samples: list[TrainingSample]
    episodic_reward: float
    is_terminal: bool
    algorithm: Literal["ppo", "arm"]


class PairedSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One paired output: the original TrainingSample with advantages now set,
    plus the corresponding AdvantageTrainingSample."""

    llm_sample: TrainingSample
    advantage_sample: AdvantageTrainingSample


class ComputeAdvantagesResponse(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """Response body. `paired_samples` is in matching order with the request's
    `samples`; one entry per input sample."""

    paired_samples: list[PairedSample]


# ---------------------------------------------------------------------------
# Protocol (typing) -- both AdvantageServerClient and FakeAdvantageServer
# implement this so the orchestrator can be parameterized over either.
# ---------------------------------------------------------------------------


class AdvantageServerClientProtocol(Protocol):
    async def compute_advantages_and_targets(
        self,
        samples: list[TrainingSample],
        episodic_reward: float,
        is_terminal: bool,
        algorithm: Literal["ppo", "arm"],
    ) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
        ...

    async def aclose(self) -> None:
        ...

    async def wait_for_ready(self, timeout: float = 600.0, poll_interval: float = 1.0) -> None:
        ...

    async def wait_for_weight_step(
        self,
        min_step: int,
        timeout: float = 3600.0,
        poll_interval: float = 1.0,
    ) -> int:
        ...


# ---------------------------------------------------------------------------
# Real HTTP client (httpx async)
# ---------------------------------------------------------------------------


_ENDPOINT_PATH = "/compute_advantages_and_targets"


class AdvantageServerClient:
    """Async HTTP client for the Advantage Server.

    Mirrors the request/response shape of the FastAPI endpoint defined in
    Phase 6b. Uses msgspec.msgpack for the body encoding (fast, compact;
    matches the rest of prime-rl's transport stack). Falls back to JSON if
    the server only accepts JSON -- the encoding is selected per-instance.
    """

    def __init__(self, config: AdvantageServerClientConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout),
        )

    async def compute_advantages_and_targets(
        self,
        samples: list[TrainingSample],
        episodic_reward: float,
        is_terminal: bool,
        algorithm: Literal["ppo", "arm"],
    ) -> list[tuple[TrainingSample, AdvantageTrainingSample]]:
        """POST one rollout's samples + metadata; receive paired outputs."""
        request = ComputeAdvantagesRequest(
            samples=samples,
            episodic_reward=episodic_reward,
            is_terminal=is_terminal,
            algorithm=algorithm,
        )
        body = msgspec.msgpack.encode(request)
        response = await self._client.post(
            _ENDPOINT_PATH,
            content=body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        response.raise_for_status()
        decoded = msgspec.msgpack.decode(response.content, type=ComputeAdvantagesResponse)
        return [(p.llm_sample, p.advantage_sample) for p in decoded.paired_samples]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def wait_for_ready(self, timeout: float = 600.0, poll_interval: float = 1.0) -> None:
        """Poll the server's /health endpoint until it returns 200 OK.

        Mirrors the orchestrator's existing inference-pool readiness pattern.
        Connection-refused errors during the first few seconds (server hasn't
        bound the socket yet) are caught and retried; the timeout ceiling
        protects against a permanently-unreachable server. Raises
        TimeoutError on timeout.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        last_error: str | None = None
        while loop.time() < deadline:
            try:
                response = await self._client.get("/health")
                if response.status_code == 200:
                    return
                last_error = f"http {response.status_code}"
            except Exception as e:  # noqa: BLE001 -- surface any transport error in the timeout message
                last_error = type(e).__name__ + ": " + str(e)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"Advantage Server at {self._config.base_url} did not become ready "
            f"within {timeout}s. Last error: {last_error}"
        )

    async def wait_for_weight_step(
        self,
        min_step: int,
        timeout: float = 3600.0,
        poll_interval: float = 1.0,
    ) -> int:
        """Block until the AdvServer's weight_step >= min_step.

        Polls GET /weight_status. Used by the orchestrator to enforce the
        AdvTrainer<->orchestrator lockstep invariant: before issuing step N's
        /compute_advantages_and_targets calls, the orchestrator confirms the
        AdvServer has applied the AdvTrainer's broadcast for batch N-1
        (which advances weight_step from N-1 to N).

        Returns the AdvServer's weight_step at the moment the condition was
        satisfied (useful for diagnostics / drift logging). Returns
        immediately when the condition is already satisfied at first poll.

        The timeout default (1 hour) is generous because the AdvTrainer's
        Jin-aligned inner loop can take 10+ minutes per cycle at ALFWorld
        scale. Hitting it indicates the AdvTrainer is hung or has crashed --
        failing the run is correct.

        Args:
            min_step: required minimum value of AdvServer's weight_step
                counter. Pass 0 to short-circuit the wait (always satisfied).
            timeout: maximum time to wait in seconds. Raises TimeoutError
                if exceeded.
            poll_interval: seconds between polls of /weight_status.
        """
        import asyncio

        if min_step <= 0:
            # weight_step starts at 0 at app construction; condition is
            # always satisfied. No HTTP call needed.
            return 0

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        last_seen: int | None = None
        last_error: str | None = None
        while loop.time() < deadline:
            try:
                response = await self._client.get("/weight_status")
                if response.status_code == 200:
                    payload = msgspec.json.decode(response.content)
                    last_seen = int(payload["weight_step"])
                    if last_seen >= min_step:
                        return last_seen
                else:
                    last_error = f"http {response.status_code}"
            except Exception as e:  # noqa: BLE001
                last_error = type(e).__name__ + ": " + str(e)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"Advantage Server at {self._config.base_url} did not reach "
            f"weight_step >= {min_step} within {timeout}s. "
            f"Last observed weight_step: {last_seen}. Last error: {last_error}. "
            f"AdvTrainer may be hung or crashed."
        )
