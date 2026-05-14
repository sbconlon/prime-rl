"""Configuration for the Advantage Server process and its orchestrator-side client.

Two configs:
- `AdvantageServerConfig`: consumed by `prime_rl.advantage_server.server.main` --
  describes the value-network backbone, LoRA hyperparameters, Polyak tau, and
  the HTTP server's host/port.
- `AdvantageServerClientConfig`: consumed by the orchestrator -- describes how
  to reach the server (base URL + request timeout).

Phase 6 only wires the configs and the compute / client / fake; the orchestrator
integration and the actual HTTP server entrypoint land in Phase 6b.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.utils.config import BaseConfig


class AdvantageServerConfig(BaseConfig):
    """Server-side configuration for the Advantage Server process.

    Loaded once at startup; the server constructs ValueNetworkBackbone using
    these fields and then serves the /compute_advantages_and_targets endpoint.
    """

    model_name: Annotated[
        str,
        Field(
            description=(
                "Base model name or path for the value networks. Production: SFT "
                "checkpoint path so V/Q+/V_target start from a meaningful frozen "
                "encoder. Tests use a tiny Qwen2 config-built model."
            ),
        ),
    ]

    lora: Annotated[
        LoRAConfig,
        Field(
            description=(
                "LoRA hyperparameters for the value networks (rank, alpha, dropout, "
                "target modules). Phase 4 commits to rank=16, alpha=32, dropout=0 "
                "as defaults; ablations available without code changes."
            ),
        ),
    ] = LoRAConfig()

    polyak_tau: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Polyak averaging rate for V_target tracking V. Phase 4 commits to "
                "0.005 (SAC default); lower values stabilize at the cost of slower "
                "tracking."
            ),
        ),
    ] = 0.005

    host: Annotated[
        str,
        Field(description="HTTP server host. Use '0.0.0.0' for cross-node access; '127.0.0.1' for local-only."),
    ] = "127.0.0.1"

    port: Annotated[
        int,
        Field(ge=1024, le=65535, description="HTTP server port for the /compute_advantages_and_targets endpoint."),
    ] = 8200

    n_step: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "n-step return horizon for ARM v_target / Q+ accumulation "
                "(see Phase 3). Forwarded to arm_regret_matching_advantage_fn. "
                "Ignored for PPO."
            ),
        ),
    ] = 5

    gamma: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Discount factor passed to PPO GAE / ARM n-step return.",
        ),
    ] = 0.99

    lam: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="GAE lambda (PPO only). Ignored for ARM.",
        ),
    ] = 0.95

    arm_phi_decay: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "CFR+ accumulation decay. phi = arm_phi_decay * max(0, "
                "Q+_prev - V_prev). gamma=1.0 (default) preserves original "
                "behavior where regret accumulates unbounded; gamma<1 bounds "
                "Q+ at steady-state. Typical: 0.9 (10%% per-iteration decay). "
                "Q+_eq = (g - gamma * V) / (1 - gamma) when gamma < 1."
            ),
        ),
    ] = 1.0

    arm_advantage_formula: Annotated[
        Literal["regret_matching", "log_regret_ratio"],
        Field(
            description=(
                "Which ARM LLM-advantage formula to use. "
                "\"regret_matching\" (default, original Phase 3 behavior): "
                "A(a*) = p_RM(a*) - 1/K. p_RM never shrinks just because the "
                "inference policy already matches it, allowing a runaway-positive-"
                "advantage feedback loop. "
                "\"log_regret_ratio\" (2026-05-13 follow-up): "
                "A(a*) = log(p_RM(a*)) - log(pi_inference(a*)). Stabilizing: "
                "advantage approaches 0 as pi_inference -> p_RM, and goes "
                "negative when pi_inference overshoots p_RM. Closer in spirit "
                "to CFR\'s policy-tracking-target semantics."
            ),
        ),
    ] = "regret_matching"


class AdvantageServerClientConfig(BaseConfig):
    """Orchestrator-side configuration for reaching the Advantage Server.

    Required when `OrchestratorConfig.algorithm in {"ppo", "arm"}`; ignored
    otherwise.
    """

    base_url: Annotated[
        str,
        Field(
            description=(
                "Base URL of the Advantage Server, e.g. 'http://127.0.0.1:8200'. "
                "Must match the server's host/port."
            ),
        ),
    ] = "http://127.0.0.1:8200"

    request_timeout: Annotated[
        float,
        Field(
            gt=0.0,
            description=(
                "Per-request timeout in seconds. The server's compute time grows "
                "with rollout length; for production rollouts (~5000-9000 tokens) "
                "this should be conservative until Phase 6.5's optimizations land."
            ),
        ),
    ] = 600.0
