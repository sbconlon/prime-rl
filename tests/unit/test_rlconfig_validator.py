"""Phase 6b -- RLConfig cross-validator tests for algorithm + advantage_server consistency."""

from __future__ import annotations

import pytest

from prime_rl.configs.advantage_server import (
    AdvantageServerClientConfig,
    AdvantageServerConfig,
)
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import LoRAConfig, TrainerConfig


def _make_orchestrator(algorithm: str, with_client: bool) -> OrchestratorConfig:
    return OrchestratorConfig(
        model={"name": "test"},  # type: ignore[arg-type]
        algorithm=algorithm,  # type: ignore[arg-type]
        advantage_server=AdvantageServerClientConfig() if with_client else None,
    )


def _make_trainer() -> TrainerConfig:
    return TrainerConfig(model={"name": "test"})  # type: ignore[arg-type]


def _make_advantage_server_config() -> AdvantageServerConfig:
    return AdvantageServerConfig(
        model_name="test", lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    )


def test_grpo_no_advantage_server_passes():
    """Default GRPO setup: no launcher advantage_server, no orchestrator client."""
    cfg = RLConfig(
        trainer=_make_trainer(),
        orchestrator=_make_orchestrator("grpo", with_client=False),
    )
    assert cfg.advantage_server is None
    assert cfg.orchestrator.advantage_server is None


def test_grpo_with_launcher_advantage_server_raises():
    """GRPO with a launcher advantage_server is a misconfig."""
    with pytest.raises(ValueError, match="grpo"):
        RLConfig(
            trainer=_make_trainer(),
            orchestrator=_make_orchestrator("grpo", with_client=False),
            advantage_server=_make_advantage_server_config(),
        )


def test_ppo_with_launcher_advantage_server_passes():
    """PPO with a launcher advantage_server (and no orchestrator client) passes
    the cross-validator (the launcher will spawn the subprocess).

    This config also needs the orchestrator-side client config because the
    OrchestratorConfig validator (Phase 6a) requires it. Set both.
    """
    cfg = RLConfig(
        trainer=_make_trainer(),
        orchestrator=_make_orchestrator("ppo", with_client=True),
        advantage_server=_make_advantage_server_config(),
    )
    assert cfg.advantage_server is not None


def test_ppo_with_orchestrator_client_only_passes():
    """PPO with only the orchestrator-side client (external Advantage Server)
    passes the cross-validator -- the launcher won't spawn a subprocess but
    the orchestrator can reach an external server."""
    cfg = RLConfig(
        trainer=_make_trainer(),
        orchestrator=_make_orchestrator("ppo", with_client=True),
        advantage_server=None,
    )
    assert cfg.advantage_server is None
    assert cfg.orchestrator.advantage_server is not None


def test_ppo_with_neither_raises():
    """PPO with neither launcher server nor orchestrator client is a misconfig.

    Note: this is caught by OrchestratorConfig's existing validator (Phase 6a)
    before RLConfig's cross-validator. Either error message is acceptable;
    the test just asserts SOMETHING blocks the misconfig.
    """
    with pytest.raises(ValueError, match="advantage_server"):
        RLConfig(
            trainer=_make_trainer(),
            orchestrator=_make_orchestrator("ppo", with_client=False),
            advantage_server=None,
        )


def test_arm_with_both_launcher_server_and_orchestrator_client_passes():
    """The most explicit config: launcher spawns the server, orchestrator points
    its client at the launcher's host/port."""
    cfg = RLConfig(
        trainer=_make_trainer(),
        orchestrator=_make_orchestrator("arm", with_client=True),
        advantage_server=_make_advantage_server_config(),
    )
    assert cfg.advantage_server is not None
    assert cfg.orchestrator.advantage_server is not None
