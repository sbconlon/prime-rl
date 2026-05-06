"""Phase 6a -- OrchestratorConfig validation for the algorithm + advantage_server fields."""

from __future__ import annotations

import pytest

from prime_rl.configs.advantage_server import AdvantageServerClientConfig
from prime_rl.configs.orchestrator import OrchestratorConfig


def test_default_orchestrator_config_uses_grpo_no_advantage_server():
    """Default OrchestratorConfig has algorithm='grpo' and no advantage_server."""
    config = OrchestratorConfig(model={"name": "test-model"})  # type: ignore[arg-type]
    assert config.algorithm == "grpo"
    assert config.advantage_server is None


def test_grpo_with_advantage_server_set_is_allowed():
    """algorithm='grpo' with advantage_server set is technically allowed; the
    advantage_server config is just ignored."""
    config = OrchestratorConfig(
        model={"name": "test-model"},  # type: ignore[arg-type]
        algorithm="grpo",
        advantage_server=AdvantageServerClientConfig(),
    )
    assert config.algorithm == "grpo"
    # Validation passes -- no error raised.


def test_ppo_without_advantage_server_raises():
    """algorithm='ppo' without advantage_server config triggers a validator error."""
    with pytest.raises(ValueError, match="advantage_server"):
        OrchestratorConfig(
            model={"name": "test-model"},  # type: ignore[arg-type]
            algorithm="ppo",
            advantage_server=None,
        )


def test_arm_without_advantage_server_raises():
    """algorithm='arm' without advantage_server config triggers a validator error."""
    with pytest.raises(ValueError, match="advantage_server"):
        OrchestratorConfig(
            model={"name": "test-model"},  # type: ignore[arg-type]
            algorithm="arm",
            advantage_server=None,
        )


def test_ppo_with_advantage_server_passes():
    """algorithm='ppo' with advantage_server config validates cleanly."""
    config = OrchestratorConfig(
        model={"name": "test-model"},  # type: ignore[arg-type]
        algorithm="ppo",
        advantage_server=AdvantageServerClientConfig(base_url="http://localhost:8200"),
    )
    assert config.algorithm == "ppo"
    assert config.advantage_server is not None
    assert config.advantage_server.base_url == "http://localhost:8200"


def test_arm_with_advantage_server_passes():
    config = OrchestratorConfig(
        model={"name": "test-model"},  # type: ignore[arg-type]
        algorithm="arm",
        advantage_server=AdvantageServerClientConfig(),
    )
    assert config.algorithm == "arm"
    assert config.advantage_server is not None
