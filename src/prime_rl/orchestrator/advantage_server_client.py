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
