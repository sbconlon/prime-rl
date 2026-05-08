"""Phase 7b: parallel transport classes for AdvantageTrainingBatch.

Same structure as the existing TrainingBatch transport (orchestrator -> LLM
Trainer); new channel for the orchestrator -> Advantage Trainer flow.

The receivers subclass the existing FileSystem / ZMQ TrainingBatch receivers
and override `_batch_type` (added as a parameterizable class attr to
`transport/base.py` in Phase 7b). The senders are essentially aliases (the
underlying msgspec encoder is type-agnostic; only the type hint changes).

Wire format is identical in shape to TrainingBatch: a list of samples + step
+ run_idx. The samples differ (AdvantageTrainingSample carries v_targets and
q_plus_targets instead of advantages), so the encoded bytes differ -- the
receiver's decoder type is what makes them decode correctly.
"""

from __future__ import annotations

from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transport.filesystem import (
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.types import AdvantageTrainingBatch
from prime_rl.transport.zmq import (
    ZMQTrainingBatchReceiver,
    ZMQTrainingBatchSender,
)


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


class FileSystemAdvantageTrainingBatchSender(FileSystemTrainingBatchSender):
    """Filesystem-based sender for AdvantageTrainingBatch.

    Inherits the parent's encode + write-to-disk implementation (msgspec
    encoder is type-agnostic). Override only the type hint on `send`.
    """

    def send(self, batch: AdvantageTrainingBatch) -> None:  # type: ignore[override]
        super().send(batch)  # type: ignore[arg-type]


class FileSystemAdvantageTrainingBatchReceiver(FileSystemTrainingBatchReceiver):
    """Filesystem-based receiver for AdvantageTrainingBatch.

    Inherits the parent's read-from-disk + decode implementation; override
    `_batch_type` so the parent's `__init__` sets up a decoder for
    AdvantageTrainingBatch instead of TrainingBatch.
    """

    _batch_type = AdvantageTrainingBatch

    def receive(self) -> list[AdvantageTrainingBatch]:  # type: ignore[override]
        return super().receive()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------


class ZMQAdvantageTrainingBatchSender(ZMQTrainingBatchSender):
    """ZMQ-based sender for AdvantageTrainingBatch."""

    def send(self, batch: AdvantageTrainingBatch) -> None:  # type: ignore[override]
        super().send(batch)  # type: ignore[arg-type]


class ZMQAdvantageTrainingBatchReceiver(ZMQTrainingBatchReceiver):
    """ZMQ-based receiver for AdvantageTrainingBatch."""

    _batch_type = AdvantageTrainingBatch

    def receive(self) -> list[AdvantageTrainingBatch]:  # type: ignore[override]
        return super().receive()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Setup helpers (mirror prime_rl.transport.setup_training_batch_*)
# ---------------------------------------------------------------------------


def setup_advantage_training_batch_sender(
    output_dir: Path, transport: TransportConfig
) -> FileSystemAdvantageTrainingBatchSender | ZMQAdvantageTrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemAdvantageTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        return ZMQAdvantageTrainingBatchSender(output_dir, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_advantage_training_batch_receiver(
    transport: TransportConfig,
) -> FileSystemAdvantageTrainingBatchReceiver | ZMQAdvantageTrainingBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemAdvantageTrainingBatchReceiver()
    elif transport.type == "zmq":
        return ZMQAdvantageTrainingBatchReceiver(transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")
