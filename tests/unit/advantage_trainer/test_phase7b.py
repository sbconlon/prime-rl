"""Phase 7b tests: transport round-trip + data adapter + _train_step + config parse."""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest
import torch
import torch.distributed as dist
from torch import optim
from transformers import Qwen2Config, Qwen2Model

from prime_rl.advantage_trainer.data import (
    PreparedAdvantageInputs,
    prepare_advantage_sample,
)
from prime_rl.advantage_trainer.train import _train_step
from prime_rl.configs.advantage_trainer import AdvantageTrainerConfig
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone, ValueNetworkConfig
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


@pytest.fixture(autouse=True, scope="module")
def _init_process_group():
    """Filesystem transport receivers require torch.distributed to be initialized
    (the multi-run manager uses dist primitives). Set up Gloo for the test
    module's lifetime, then tear down."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method="tcp://localhost:12378", rank=0, world_size=1
        )
        yield
        dist.destroy_process_group()
    else:
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_advantage_sample(
    completion_len: int = 3,
    *,
    with_q_plus: bool = False,
    completion_mask: list[bool] | None = None,
) -> AdvantageTrainingSample:
    """Construct a synthetic AdvantageTrainingSample for tests."""
    if completion_mask is None:
        completion_mask = [True] * completion_len
    return AdvantageTrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + completion_len)),
        completion_mask=completion_mask,
        v_targets=[float(i) * 0.1 for i in range(completion_len)],
        q_plus_targets=(
            [float(i) * 0.2 for i in range(completion_len)] if with_q_plus else None
        ),
    )


def _setup_multi_run_manager(tmp_path: Path) -> None:
    """Filesystem transport receivers require a MultiRunManager. Set one up
    pointing at a tmp output dir for tests."""
    import prime_rl.trainer.runs as runs
    from prime_rl.trainer.runs import setup_multi_run_manager
    from prime_rl.trainer.world import reset_world

    reset_world()
    runs._MULTI_RUN_MANAGER = None
    setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))


def _tiny_backbone() -> ValueNetworkBackbone:
    """Tiny 2-layer Qwen2 ValueNetworkBackbone with non-zero LoRA + heads.

    Production zero-init: lora_B = zeros AND head.weight = zeros. Both make
    the V/Q+ output zero AND the gradient back to lora_A zero (since
    dL/dlora_A passes through lora_B.T which is zero). Tests need both
    LoRA tensors and heads non-zero so gradients flow to all trainable
    params. Mirrors the Phase 7a helper in test_routing_polyak_step.py.
    """
    from prime_rl.orchestrator.value_networks import (
        Q_PLUS_SLOT,
        V_SLOT,
        V_TARGET_SLOT,
    )
    from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    with torch.no_grad():
        # Break LoRA zero-init at all three slots.
        for module in backbone.base_model.modules():
            if isinstance(module, MultiLoRALinear):
                module.lora_A[V_SLOT].fill_(0.4)
                module.lora_B[V_SLOT].fill_(0.4)
                module.lora_A[Q_PLUS_SLOT].fill_(0.6)
                module.lora_B[Q_PLUS_SLOT].fill_(0.6)
                module.lora_A[V_TARGET_SLOT].fill_(0.5)
                module.lora_B[V_TARGET_SLOT].fill_(0.5)
        # Break head zero-init.
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


# ---------------------------------------------------------------------------
# Transport round-trip
# ---------------------------------------------------------------------------


def test_advantage_training_batch_msgspec_round_trip():
    """msgspec encode + decode round-trip preserves AdvantageTrainingBatch.

    Verifies the wire format. The actual filesystem-write-and-read path
    is exercised by the existing FileSystem*TrainingBatch* tests in
    test_packer.py (which my parallel classes inherit unchanged from).
    """
    sample = _make_advantage_sample(completion_len=3, with_q_plus=True)
    batch = AdvantageTrainingBatch(examples=[sample], step=0, run_idx=0)

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=AdvantageTrainingBatch)

    encoded = encoder.encode(batch)
    decoded = decoder.decode(encoded)

    # Re-encode both and compare bytes for full round-trip equality.
    assert encoder.encode(decoded) == encoded
    # Sanity-check semantic fields.
    assert isinstance(decoded, AdvantageTrainingBatch)
    assert decoded.step == batch.step
    assert len(decoded.examples) == 1
    got_sample = decoded.examples[0]
    assert got_sample.prompt_ids == sample.prompt_ids
    assert got_sample.completion_ids == sample.completion_ids
    assert got_sample.v_targets == sample.v_targets
    assert got_sample.q_plus_targets == sample.q_plus_targets


def test_advantage_training_batch_receiver_uses_correct_decoder_type(tmp_path: Path):
    """Phase 7b's parallel receiver classes override _batch_type so the
    msgspec decoder is configured for AdvantageTrainingBatch, not
    TrainingBatch. Confirms the inheritance + class-attr override works."""
    _setup_multi_run_manager(tmp_path)

    receiver = FileSystemAdvantageTrainingBatchReceiver()
    assert receiver.decoder.type is AdvantageTrainingBatch
    # Also via the setup helper.
    receiver2 = setup_advantage_training_batch_receiver(FileSystemTransportConfig())
    assert receiver2.decoder.type is AdvantageTrainingBatch


def test_setup_helpers_dispatch_to_filesystem(tmp_path: Path):
    """setup_advantage_training_batch_{sender,receiver} return filesystem
    classes for FileSystemTransportConfig."""
    _setup_multi_run_manager(tmp_path)

    transport = FileSystemTransportConfig()
    sender = setup_advantage_training_batch_sender(tmp_path, transport)
    receiver = setup_advantage_training_batch_receiver(transport)
    assert isinstance(sender, FileSystemAdvantageTrainingBatchSender)
    assert isinstance(receiver, FileSystemAdvantageTrainingBatchReceiver)


def test_existing_training_batch_receiver_unaffected_by_refactor():
    """The 1-line refactor to transport/base.py adds a class attr without
    changing behavior of the existing TrainingBatchReceiver subclasses."""
    from prime_rl.transport.base import TrainingBatchReceiver
    from prime_rl.transport.types import TrainingBatch

    # The default _batch_type is TrainingBatch (preserves old behavior).
    assert TrainingBatchReceiver._batch_type is TrainingBatch


# ---------------------------------------------------------------------------
# data adapter
# ---------------------------------------------------------------------------


def test_prepare_advantage_sample_aligns_targets_to_full_input():
    """Prompt+completion shape; v_targets/q_plus_targets aligned with zeros at
    prompt positions; loss_mask=False at prompt + bridge positions."""
    sample = _make_advantage_sample(completion_len=3, with_q_plus=True)
    prepared = prepare_advantage_sample(sample)

    prompt_len = len(sample.prompt_ids)
    completion_len = len(sample.completion_ids)
    full_len = prompt_len + completion_len

    assert prepared.input_ids.shape == (1, full_len)
    assert prepared.v_targets.shape == (full_len,)
    assert prepared.q_plus_targets is not None
    assert prepared.q_plus_targets.shape == (full_len,)
    assert prepared.loss_mask.shape == (full_len,)

    # Prompt positions: targets zero, loss_mask False.
    assert prepared.v_targets[:prompt_len].abs().max().item() == 0.0
    assert prepared.q_plus_targets[:prompt_len].abs().max().item() == 0.0
    assert (~prepared.loss_mask[:prompt_len]).all().item()

    # Completion positions: targets match the sample's, loss_mask True (since
    # all completion_mask entries are True in this test). Use pytest.approx
    # to absorb float32 conversion epsilon.
    assert prepared.v_targets[prompt_len:].tolist() == pytest.approx(
        sample.v_targets, abs=1e-6
    )
    assert prepared.q_plus_targets[prompt_len:].tolist() == pytest.approx(
        sample.q_plus_targets, abs=1e-6
    )
    assert prepared.loss_mask[prompt_len:].all().item()


def test_prepare_advantage_sample_loss_mask_excludes_bridge_positions():
    """Multi-turn rollouts have completion_mask=False at bridge tokens.
    The data adapter's loss_mask must be False there."""
    completion_mask = [True, True, False, False, True]  # bridge at positions 2,3
    sample = _make_advantage_sample(
        completion_len=5, with_q_plus=False, completion_mask=completion_mask
    )
    prepared = prepare_advantage_sample(sample)

    prompt_len = len(sample.prompt_ids)
    completion_loss_mask = prepared.loss_mask[prompt_len:].tolist()
    assert completion_loss_mask == completion_mask


def test_prepare_advantage_sample_ppo_omits_q_plus():
    """PPO sample has q_plus_targets=None; prepared.q_plus_targets should be None."""
    sample = _make_advantage_sample(completion_len=3, with_q_plus=False)
    prepared = prepare_advantage_sample(sample)
    assert prepared.q_plus_targets is None


def test_prepare_advantage_sample_validates_target_lengths():
    """Wrong v_targets length -> raises informative error."""
    sample = AdvantageTrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[10, 11, 12],
        completion_mask=[True, True, True],
        v_targets=[0.1, 0.2],  # wrong length (2 instead of 3)
        q_plus_targets=None,
    )
    with pytest.raises(ValueError, match="v_targets length"):
        prepare_advantage_sample(sample)


# ---------------------------------------------------------------------------
# _train_step (testable in isolation, no transport)
# ---------------------------------------------------------------------------


def test_train_step_completes_ppo():
    """Single PPO _train_step on a tiny backbone: returns metrics with finite
    loss; V's adapter moved (gradient descent fired)."""
    from prime_rl.orchestrator.value_networks import V_SLOT
    from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear

    backbone = _tiny_backbone()
    backbone.train()
    # Snapshot V's LoRA before.
    v_lora_before = []
    for module in backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            v_lora_before.append(module.lora_A[V_SLOT].detach().clone())

    trainable = [p for p in backbone.parameters() if p.requires_grad]
    # Adam @ 1e-2 produces visible early-step updates (Phase 7a's
    # test_polyak_fires_after_each_step uses the same recipe).
    optimizer = optim.Adam(trainable, lr=1e-2)

    sample = _make_advantage_sample(completion_len=3, with_q_plus=False)
    batch = AdvantageTrainingBatch(examples=[sample], step=0, run_idx=0)

    metrics = _train_step(
        backbone=backbone,
        optimizer=optimizer,
        batch=batch,
        algorithm="ppo",
        polyak_tau=0.005,
    )

    assert "mean_loss" in metrics
    assert metrics["n_samples"] == 1
    import math
    assert math.isfinite(metrics["mean_loss"])
    assert metrics["l_q"] == 0.0  # PPO -> no Q+ contribution

    # V's adapter changed (gradient descent fired).
    v_changed = False
    for i, module in enumerate(
        m for m in backbone.base_model.modules() if isinstance(m, MultiLoRALinear)
    ):
        if not torch.allclose(module.lora_A[V_SLOT], v_lora_before[i], atol=1e-9):
            v_changed = True
            break
    assert v_changed, "V's adapter did not move during _train_step"


def test_train_step_completes_arm_with_multi_sample_batch():
    """ARM _train_step with a 2-sample batch returns aggregated metrics."""
    backbone = _tiny_backbone()
    backbone.train()
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-2)

    s0 = _make_advantage_sample(completion_len=3, with_q_plus=True)
    s1 = _make_advantage_sample(completion_len=2, with_q_plus=True)
    batch = AdvantageTrainingBatch(examples=[s0, s1], step=0, run_idx=0)

    metrics = _train_step(
        backbone=backbone,
        optimizer=optimizer,
        batch=batch,
        algorithm="arm",
        polyak_tau=0.005,
    )

    assert metrics["n_samples"] == 2
    assert metrics["l_v"] >= 0.0
    assert metrics["l_q"] >= 0.0


def test_train_step_empty_batch_returns_zero():
    """An empty batch shouldn't crash and shouldn't perform updates."""
    backbone = _tiny_backbone()
    backbone.train()
    optimizer = optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad], lr=1e-2
    )
    empty_batch = AdvantageTrainingBatch(examples=[], step=0, run_idx=0)
    metrics = _train_step(
        backbone=backbone,
        optimizer=optimizer,
        batch=empty_batch,
        algorithm="ppo",
        polyak_tau=0.005,
    )
    assert metrics["n_samples"] == 0


# ---------------------------------------------------------------------------
# Config-parse smoke
# ---------------------------------------------------------------------------


def test_advantage_trainer_config_constructs_with_defaults():
    """AdvantageTrainerConfig requires explicit model + algorithm; the rest
    have defaults."""
    config = AdvantageTrainerConfig(
        model=ValueNetworkConfig(
            base_model_name="test-model",
            lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        ),
        algorithm="ppo",
    )
    assert config.algorithm == "ppo"
    assert config.polyak_tau == 0.005
    assert config.max_steps == 1000
    assert config.transport.type == "filesystem"


def test_advantage_trainer_config_arm_with_zmq_transport():
    """Sanity-check ARM + ZMQ config construction."""
    from prime_rl.configs.shared import ZMQTransportConfig

    config = AdvantageTrainerConfig(
        model=ValueNetworkConfig(
            base_model_name="test-model",
            lora=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        ),
        algorithm="arm",
        transport=ZMQTransportConfig(),
    )
    assert config.algorithm == "arm"
    assert config.transport.type == "zmq"
