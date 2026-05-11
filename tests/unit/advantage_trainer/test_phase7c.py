"""Phase 7c tests: integration plumbing for the Advantage Trainer.

Coverage:
  1. Path-based filesystem transport: send/receive round-trip and edge cases.
  2. Weight broadcast: serializer round-trip, /update_weights endpoint
     contract, and the Trainer-side _broadcast_weights helper (with a
     mocked httpx transport).
  3. Launch script: write_subconfigs renders advantage_trainer.toml with
     launcher-wired transport_input_dir and advantage_server_url.

The 4-process pipeline integration test (orchestrator + trainer + server
running together) is cluster-only and batched into Phase 10 e2e prep.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import httpx
import msgspec
import pytest
import torch
import tomli
from fastapi.testclient import TestClient
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_server.server import create_app
from prime_rl.advantage_trainer.train import (
    _broadcast_weights,
    _serialize_trainable_state_dict,
)
from prime_rl.configs.advantage_server import (
    AdvantageServerClientConfig,
    AdvantageServerConfig,
)
from prime_rl.configs.advantage_trainer import AdvantageTrainerConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.shared import (
    FileSystemTransportConfig,
    ZMQTransportConfig,
)
from prime_rl.configs.trainer import LoRAConfig, TrainerConfig
from prime_rl.entrypoints.rl import write_subconfigs
from prime_rl.orchestrator.value_networks import (
    Q_PLUS_SLOT,
    V_SLOT,
    V_TARGET_SLOT,
    ValueNetworkBackbone,
    ValueNetworkConfig,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.transport.advantage_batch import (
    FileSystemAdvantageTrainingBatchReceiver,
    FileSystemAdvantageTrainingBatchSender,
    setup_advantage_training_batch_receiver,
    setup_advantage_training_batch_sender,
)
from prime_rl.transport.types import (
    AdvantageTrainingBatch,
    AdvantageTrainingSample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def _tiny_backbone(seed: int = 0) -> ValueNetworkBackbone:
    """Tiny 2-layer Qwen2 ValueNetworkBackbone with non-zero LoRA + heads."""
    torch.manual_seed(seed)
    backbone = ValueNetworkBackbone(
        Qwen2Model(_tiny_qwen2_config()),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    with torch.no_grad():
        # Break LoRA zero-init at all three slots so the trainable state
        # dict has non-zero values to round-trip.
        for module in backbone.base_model.modules():
            if isinstance(module, MultiLoRALinear):
                module.lora_A[V_SLOT].fill_(0.4 + 0.01 * seed)
                module.lora_B[V_SLOT].fill_(0.4 + 0.01 * seed)
                module.lora_A[Q_PLUS_SLOT].fill_(0.6 + 0.01 * seed)
                module.lora_B[Q_PLUS_SLOT].fill_(0.6 + 0.01 * seed)
                module.lora_A[V_TARGET_SLOT].fill_(0.5 + 0.01 * seed)
                module.lora_B[V_TARGET_SLOT].fill_(0.5 + 0.01 * seed)
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def _make_advantage_sample(completion_len: int = 3) -> AdvantageTrainingSample:
    """Build a minimal AdvantageTrainingSample for transport round-trip."""
    return AdvantageTrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=[True] * completion_len,
        v_targets=[0.0] * 2 + [float(i) for i in range(completion_len)],
        q_plus_targets=[0.0] * 2 + [float(i) * 0.5 for i in range(completion_len)],
    )


# ---------------------------------------------------------------------------
# 1. Path-based filesystem transport: send/receive round-trip
# ---------------------------------------------------------------------------


def test_filesystem_advantage_send_receive_round_trip(tmp_path: Path):
    """Sender writes; same-path receiver reads back the identical batch.

    This is the core invariant the launcher relies on: as long as the
    orchestrator points its sender at output_dir X and the trainer points
    its receiver at the same X, the bytes round-trip cleanly.
    """
    sender = FileSystemAdvantageTrainingBatchSender(tmp_path)
    receiver = FileSystemAdvantageTrainingBatchReceiver(tmp_path)

    assert receiver.can_receive() is False
    assert receiver.receive() == []

    sample = _make_advantage_sample(completion_len=3)
    batch = AdvantageTrainingBatch(examples=[sample], step=0, run_idx=0)
    sender.send(batch)

    assert receiver.can_receive() is True
    received = receiver.receive()
    assert len(received) == 1
    got = received[0]
    assert isinstance(got, AdvantageTrainingBatch)
    assert got.step == 0
    assert len(got.examples) == 1
    got_sample = got.examples[0]
    assert got_sample.prompt_ids == sample.prompt_ids
    assert got_sample.completion_ids == sample.completion_ids
    assert got_sample.v_targets == sample.v_targets
    assert got_sample.q_plus_targets == sample.q_plus_targets


def test_filesystem_receiver_advances_step_and_does_not_double_read(tmp_path: Path):
    """Once a batch is consumed, can_receive() goes False until the next
    step\'s file appears -- no duplicate reads."""
    sender = FileSystemAdvantageTrainingBatchSender(tmp_path)
    receiver = FileSystemAdvantageTrainingBatchReceiver(tmp_path)

    sender.send(AdvantageTrainingBatch(examples=[_make_advantage_sample()], step=0))
    assert len(receiver.receive()) == 1
    assert receiver.can_receive() is False
    assert receiver.receive() == []

    sender.send(AdvantageTrainingBatch(examples=[_make_advantage_sample()], step=1))
    received = receiver.receive()
    assert len(received) == 1
    assert received[0].step == 1


def test_filesystem_receiver_catches_up_multiple_steps(tmp_path: Path):
    """If the sender writes faster than the receiver polls, a single
    receive() call returns all buffered batches in step order."""
    sender = FileSystemAdvantageTrainingBatchSender(tmp_path)
    receiver = FileSystemAdvantageTrainingBatchReceiver(tmp_path)

    for step in range(3):
        sender.send(AdvantageTrainingBatch(examples=[_make_advantage_sample()], step=step))

    received = receiver.receive()
    assert [b.step for b in received] == [0, 1, 2]


def test_setup_helper_requires_input_dir_for_filesystem():
    """Filesystem transport requires input_dir; setup helper enforces this
    so the launcher misconfig is caught at startup, not at first receive."""
    with pytest.raises(ValueError, match="input_dir"):
        setup_advantage_training_batch_receiver(FileSystemTransportConfig())


def test_setup_helper_filesystem_with_input_dir_succeeds(tmp_path: Path):
    receiver = setup_advantage_training_batch_receiver(
        FileSystemTransportConfig(), input_dir=tmp_path
    )
    assert isinstance(receiver, FileSystemAdvantageTrainingBatchReceiver)


# ---------------------------------------------------------------------------
# 2. Weight broadcast: serializer + endpoint + helper
# ---------------------------------------------------------------------------


def test_serialize_trainable_state_dict_includes_lora_and_heads_only():
    """Only requires_grad params are included. Frozen base params are
    excluded so the wire payload stays small (LoRA + heads only)."""
    backbone = _tiny_backbone(seed=0)
    body = _serialize_trainable_state_dict(backbone)
    state = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)

    expected = {
        name for name, p in backbone.named_parameters() if p.requires_grad
    }
    assert set(state.keys()) == expected
    # Sanity: at least lora_A/lora_B + value heads are present.
    assert any("lora_A" in k for k in state.keys())
    assert any("lora_B" in k for k in state.keys())
    assert any("v_head" in k for k in state.keys())
    assert any("q_plus_head" in k for k in state.keys())


def _build_tiny_server_app(monkeypatch: pytest.MonkeyPatch):
    """Build the FastAPI app with the tiny Qwen2 backbone preloaded.

    Mirrors test_server_endpoint._build_tiny_app: monkeypatch AutoModel so
    the lifespan startup hook builds a tiny config-driven model instead of
    downloading from HuggingFace.
    """
    from prime_rl.advantage_server import server as server_module

    class _TinyAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs):
            # AdvSrv passes dtype=bfloat16 on CUDA, dtype=float32 on CPU
            # (server.py); the tiny fake just ignores it -- the post-init
            # backbone.to(device, dtype=...) call casts anyway.
            torch.manual_seed(0)
            return Qwen2Model(_tiny_qwen2_config())

    monkeypatch.setattr(server_module, "AutoModel", _TinyAutoModel)
    config = AdvantageServerConfig(
        model_name="ignored-by-monkeypatch",
        lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
        host="127.0.0.1",
        port=8201,
        n_step=5,
    )
    return create_app(config)


def test_update_weights_endpoint_applies_state_dict(monkeypatch: pytest.MonkeyPatch):
    """POST /update_weights replaces LoRA + head weights on the live backbone.

    Round-trip:
      1. Build server with seed=0 backbone.
      2. Build a different (seed=1) backbone, serialize its trainable params.
      3. POST the bytes to /update_weights.
      4. Read a sentinel parameter on the server\'s backbone; it should
         match the seed=1 source value, not the seed=0 original.
    """
    app = _build_tiny_server_app(monkeypatch)
    new_backbone = _tiny_backbone(seed=1)
    body = _serialize_trainable_state_dict(new_backbone)

    # Pick a sentinel: the v_head linear weight (zero-initialized in the
    # production ctor; randomized in _tiny_backbone). Easy to check.
    sentinel_name = None
    for name, _ in new_backbone.named_parameters():
        if "v_head" in name and "weight" in name:
            sentinel_name = name
            break
    assert sentinel_name is not None
    new_value = dict(new_backbone.named_parameters())[sentinel_name].detach().clone()

    with TestClient(app) as client:
        # Sanity: server is up.
        assert client.get("/health").status_code == 200

        # Pre-state on server\'s backbone differs from new_backbone\'s.
        server_backbone: ValueNetworkBackbone = app.state.backbone
        before = dict(server_backbone.named_parameters())[sentinel_name].detach().clone()
        assert not torch.allclose(before.detach().cpu().float(), new_value.detach().cpu().float())

        response = client.post(
            "/update_weights",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        after = dict(server_backbone.named_parameters())[sentinel_name].detach().clone()
        assert torch.allclose(after.detach().cpu().float(), new_value.detach().cpu().float(), atol=1e-2, rtol=1e-2)
        # weight_step counter incremented.
        assert getattr(app.state, "weight_step", 0) == 1


def test_broadcast_weights_helper_returns_true_on_success(monkeypatch: pytest.MonkeyPatch):
    """_broadcast_weights uses httpx.Client(); we mock its transport so the
    POST is intercepted and a 200 response returned. No real network."""
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body_len"] = len(request.content)
        captured["content_type"] = request.headers.get("Content-Type")
        return httpx.Response(200, json={"status": "ok"})

    mock_transport = httpx.MockTransport(_handler)

    class _PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = mock_transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("prime_rl.advantage_trainer.train.httpx.Client", _PatchedClient)

    backbone = _tiny_backbone(seed=2)
    ok = _broadcast_weights(backbone, "http://test-server:8200")
    assert ok is True
    assert captured["url"] == "http://test-server:8200/update_weights"
    assert captured["body_len"] > 0
    assert captured["content_type"] == "application/octet-stream"


def test_broadcast_weights_helper_returns_false_on_5xx(monkeypatch: pytest.MonkeyPatch):
    """Non-200 responses are non-fatal: helper logs and returns False."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"server not ready")

    mock_transport = httpx.MockTransport(_handler)

    class _PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = mock_transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("prime_rl.advantage_trainer.train.httpx.Client", _PatchedClient)

    backbone = _tiny_backbone(seed=3)
    ok = _broadcast_weights(backbone, "http://test-server:8200")
    assert ok is False


def test_broadcast_weights_helper_returns_false_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Network errors are non-fatal: helper catches the exception and returns False."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    mock_transport = httpx.MockTransport(_handler)

    class _PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = mock_transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("prime_rl.advantage_trainer.train.httpx.Client", _PatchedClient)

    backbone = _tiny_backbone(seed=4)
    ok = _broadcast_weights(backbone, "http://nonexistent:8200")
    assert ok is False


# ---------------------------------------------------------------------------
# 3. Launch script: write_subconfigs renders advantage_trainer.toml correctly
# ---------------------------------------------------------------------------


def _make_trainer_cfg() -> TrainerConfig:
    return TrainerConfig(model={"name": "test"})  # type: ignore[arg-type]


def _make_orchestrator_cfg(
    *, algorithm: str, with_client: bool
) -> OrchestratorConfig:
    """RLConfig.auto_setup_output_dir overwrites orchestrator.output_dir,
    so don\'t pass it through here -- callers set RLConfig.output_dir."""
    kwargs = dict(model={"name": "test"}, algorithm=algorithm)
    if with_client:
        kwargs["advantage_server"] = AdvantageServerClientConfig()
    return OrchestratorConfig(**kwargs)  # type: ignore[arg-type]


def _make_adv_server_cfg() -> AdvantageServerConfig:
    return AdvantageServerConfig(
        model_name="test",
        lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        host="127.0.0.1",
        port=8765,
    )


def _make_adv_trainer_cfg() -> AdvantageTrainerConfig:
    return AdvantageTrainerConfig(
        model=ValueNetworkConfig(
            base_model_name="test",
            lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
            polyak_tau=0.005,
        ),
        algorithm="arm",
    )


def test_write_subconfigs_renders_advantage_trainer_toml_with_overrides(
    tmp_path: Path,
):
    """When config.advantage_trainer is set, write_subconfigs writes
    advantage_trainer.toml and overrides:
      - transport_input_dir: matches orchestrator.output_dir/advantage_trainer_transport
      - advantage_server_url: built from advantage_server.host:port

    These are the launcher\'s contract -- the orchestrator and trainer
    can\'t agree on paths/URLs without it.
    """
    rl_out = tmp_path / "rl_out"
    cfg = RLConfig(
        trainer=_make_trainer_cfg(),
        orchestrator=_make_orchestrator_cfg(
            algorithm="arm", with_client=True
        ),
        advantage_server=_make_adv_server_cfg(),
        advantage_trainer=_make_adv_trainer_cfg(),
        output_dir=rl_out,
    )
    # RLConfig.auto_setup_output_dir sets orchestrator.output_dir =
    # <rl_out>/run_default; the trainer transport is then below that.
    expected_orch_out = rl_out / "run_default"
    assert cfg.orchestrator.output_dir == expected_orch_out

    out = tmp_path / "subconfigs"
    write_subconfigs(cfg, out)

    trainer_toml = out / "advantage_trainer.toml"
    assert trainer_toml.exists()
    rendered = tomli.loads(trainer_toml.read_text())

    expected_input = (
        expected_orch_out / "advantage_trainer_transport"
    ).as_posix()
    # tomli_w writes Path as a string; compare normalized.
    assert Path(rendered["transport_input_dir"]).as_posix() == expected_input
    assert rendered["advantage_server_url"] == "http://127.0.0.1:8765"
    # Sanity: the rest of the user-supplied fields are preserved.
    assert rendered["algorithm"] == "arm"


def test_write_subconfigs_advantage_trainer_url_omitted_when_no_server(
    tmp_path: Path,
):
    """If the launcher does NOT spawn an Advantage Server (server is None),
    advantage_server_url stays None. The PPO/ARM cross-validator requires
    EITHER launcher advantage_server OR orchestrator-side client; here we
    use the orchestrator-side client, so the trainer points nowhere.
    The trainer will then train in isolation (broadcast disabled).
    """
    rl_out = tmp_path / "rl_out"
    cfg = RLConfig(
        trainer=_make_trainer_cfg(),
        orchestrator=_make_orchestrator_cfg(
            algorithm="arm", with_client=True
        ),
        advantage_server=None,  # no launcher-spawned server
        advantage_trainer=_make_adv_trainer_cfg(),
        output_dir=rl_out,
    )

    out = tmp_path / "subconfigs"
    write_subconfigs(cfg, out)

    trainer_toml = out / "advantage_trainer.toml"
    rendered = tomli.loads(trainer_toml.read_text())
    # transport_input_dir still wired (orchestrator path is independent of server).
    assert "transport_input_dir" in rendered
    # advantage_server_url not set -> excluded from TOML by exclude_none.
    assert "advantage_server_url" not in rendered


def test_write_subconfigs_skips_advantage_trainer_when_none(tmp_path: Path):
    """If config.advantage_trainer is None (GRPO path), no toml is rendered."""
    cfg = RLConfig(
        trainer=_make_trainer_cfg(),
        orchestrator=_make_orchestrator_cfg(
            algorithm="grpo", with_client=False
        ),
        output_dir=tmp_path / "rl_out",
    )
    out = tmp_path / "subconfigs"
    write_subconfigs(cfg, out)
    assert not (out / "advantage_trainer.toml").exists()
