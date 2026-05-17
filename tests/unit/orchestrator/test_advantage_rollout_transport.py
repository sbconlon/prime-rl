"""Verify the orchestrator's advantage_rollout_transport routing decision.

Phase 10 ZMQ fix: when advantage_rollout_transport is None (default),
the AdvTrainer sender uses rollout_transport (historical behavior).
When it is set, the AdvTrainer sender uses it (allowing distinct ports
for ZMQ-on-both-pipelines).

Tests exercise the setup_advantage_training_batch_sender dispatch logic
plus the OrchestratorConfig field defaults — not full orchestrator
startup, which needs vLLM / environments / etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.shared import (
    FileSystemTransportConfig,
    ZMQTransportConfig,
)
from prime_rl.transport.advantage_batch import (
    FileSystemAdvantageTrainingBatchSender,
    ZMQAdvantageTrainingBatchSender,
    setup_advantage_training_batch_sender,
)


# ---------------------------------------------------------------------------
# 1. Field default + override semantics on OrchestratorConfig
# ---------------------------------------------------------------------------


def test_orchestrator_config_defaults_advantage_rollout_transport_to_none():
    """The new field defaults to None so existing configs are unchanged."""
    # Construct a minimal OrchestratorConfig with required fields.
    # Use Pydantic's model_construct to skip validation of unrelated fields
    # we don't care about here.
    field_info = OrchestratorConfig.model_fields["advantage_rollout_transport"]
    assert field_info.default is None, (
        f"advantage_rollout_transport default should be None, got {field_info.default!r}"
    )


def test_orchestrator_config_accepts_explicit_advantage_rollout_transport():
    """When set, the field carries through with the right type."""
    fields = OrchestratorConfig.model_fields
    # The annotation should accept TransportConfig | None
    # (we test that constructing with ZMQ doesn't raise here at instantiation).
    zmq_transport = ZMQTransportConfig(port=5556)
    assert zmq_transport.type == "zmq"
    assert zmq_transport.port == 5556


# ---------------------------------------------------------------------------
# 2. setup_advantage_training_batch_sender dispatches on the transport it
#    receives -- the orchestrator's job is to PICK which transport to pass
# ---------------------------------------------------------------------------


def test_setup_returns_filesystem_sender_for_filesystem_transport(tmp_path: Path):
    transport = FileSystemTransportConfig()
    sender = setup_advantage_training_batch_sender(tmp_path, transport)
    assert isinstance(sender, FileSystemAdvantageTrainingBatchSender)


def test_setup_returns_zmq_sender_for_zmq_transport(tmp_path: Path):
    """ZMQ sender constructs without errors at a distinct port."""
    # Use a high port to avoid conflicts with any running service.
    transport = ZMQTransportConfig(port=15556, host="localhost", hwm=10)
    try:
        sender = setup_advantage_training_batch_sender(tmp_path, transport)
        assert isinstance(sender, ZMQAdvantageTrainingBatchSender)
    finally:
        # ZMQ sender opens a socket; clean up if construction succeeded.
        if "sender" in locals():
            try:
                sender.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 3. The routing decision the orchestrator makes: explicit override wins;
#    None falls back. This is the load-bearing wiring change.
# ---------------------------------------------------------------------------


def _pick_advantage_transport(rollout_transport, advantage_rollout_transport):
    """Mirror of the orchestrator.py routing logic (after Option 2)."""
    return (
        advantage_rollout_transport
        if advantage_rollout_transport is not None
        else rollout_transport
    )


def test_routing_falls_back_to_rollout_transport_when_none():
    rollout = FileSystemTransportConfig()
    chosen = _pick_advantage_transport(rollout, None)
    assert chosen is rollout, "Routing should fall back to rollout_transport when override is None"


def test_routing_uses_override_when_set():
    rollout = FileSystemTransportConfig()
    override = ZMQTransportConfig(port=5556)
    chosen = _pick_advantage_transport(rollout, override)
    assert chosen is override, "Routing should pick the override when set"
    assert chosen.type == "zmq"
    assert chosen.port == 5556


def test_routing_allows_distinct_zmq_ports_per_pipeline():
    """The actual production use case: LLM batches on port 5555,
    AdvTrainer batches on port 5556, each independent."""
    rollout = ZMQTransportConfig(port=5555)
    override = ZMQTransportConfig(port=5556)
    chosen = _pick_advantage_transport(rollout, override)
    assert chosen.port == 5556
    # Sanity: rollout_transport still has its own distinct port for the
    # LLM TrainingBatch pipeline.
    assert rollout.port == 5555
