"""Phase 7b/7c: parallel transport classes for AdvantageTrainingBatch.

Same wire format as the existing TrainingBatch transport (orchestrator -> LLM
Trainer); separate channel for the orchestrator -> Advantage Trainer flow.

Senders inherit from the existing FileSystem / ZMQ TrainingBatchSenders --
the underlying msgspec encoder is type-agnostic, only the type hint changes.

Receivers, on the other hand, implement a flat path layout that does NOT
use the global MultiRunManager. The Advantage Trainer is single-run by
design (Phase 7 DQ2: single-GPU, single concurrent run); coupling to
MultiRunManager would require initializing the singleton in the trainer
process for no benefit. Instead, the receiver takes an explicit input_dir
and reads <input_dir>/rollouts/step_N/rollouts.bin -- exactly the flat
layout the matching FileSystemAdvantageTrainingBatchSender writes.

Wire format is identical in shape to TrainingBatch: a list of samples + step
+ run_idx. The samples differ (AdvantageTrainingSample carries v_targets and
q_plus_targets instead of advantages), so the encoded bytes differ -- the
receiver's decoder type (AdvantageTrainingBatch via _batch_type) is what
makes them decode correctly.
"""

from __future__ import annotations

from pathlib import Path
from time import time

from prime_rl.configs.shared import TransportConfig, ZMQTransportConfig
from prime_rl.transport.base import TrainingBatchReceiver
from prime_rl.transport.filesystem import (
    BATCH_FILE_NAME,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.types import AdvantageTrainingBatch
from prime_rl.transport.zmq import (
    ZMQTrainingBatchReceiver,
    ZMQTrainingBatchSender,
)
from prime_rl.utils.pathing import get_rollout_dir, get_step_path


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


class FileSystemAdvantageTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem-based receiver for AdvantageTrainingBatch (Phase 7c).

    Path-based reader (no MultiRunManager). Reads from
    <input_dir>/rollouts/step_N/rollouts.bin -- the same flat layout the
    matching FileSystemAdvantageTrainingBatchSender writes when given that
    same input_dir as its output_dir.

    Args:
      input_dir: Directory the orchestrator writes batches into. The launcher
        wires this to <orchestrator_output_dir>/advantage_trainer_transport.
      start_step: Step to begin reading at. Defaults to 0; ckpt resume can
        pass a higher value (Phase 7d).
    """

    _batch_type = AdvantageTrainingBatch

    def __init__(self, input_dir: Path, start_step: int = 0) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.rollout_dir = get_rollout_dir(input_dir)
        self._next_step = start_step
        self._waiting_since: float | None = None

    def _batch_path(self, step: int) -> Path:
        return get_step_path(self.rollout_dir, step) / BATCH_FILE_NAME

    def can_receive(self) -> bool:
        """True iff the next-expected step\'s batch file exists."""
        return self._batch_path(self._next_step).exists()

    def receive(self) -> list[AdvantageTrainingBatch]:  # type: ignore[override]
        """Read all currently-available batches in step order.

        Catches up if multiple steps have been written since the last call.
        Returns [] when nothing new is available.
        """
        now = time()
        if self.can_receive():
            self._waiting_since = None
        else:
            self._waiting_since = self._waiting_since or now
            return []

        batches: list[AdvantageTrainingBatch] = []
        while True:
            path = self._batch_path(self._next_step)
            if not path.exists():
                break
            try:
                with open(path, "rb") as f:
                    batch: AdvantageTrainingBatch = self.decoder.decode(f.read())
                batches.append(batch)
            except Exception as e:
                self.logger.error(
                    f"Error loading advantage batch from {path}: {e}"
                )
                break
            self._next_step += 1
        return batches


# ---------------------------------------------------------------------------
# ZMQ
# ---------------------------------------------------------------------------


class ZMQAdvantageTrainingBatchSender(ZMQTrainingBatchSender):
    """ZMQ-based sender for AdvantageTrainingBatch."""

    def send(self, batch: AdvantageTrainingBatch) -> None:  # type: ignore[override]
        super().send(batch)  # type: ignore[arg-type]


class ZMQAdvantageTrainingBatchReceiver(TrainingBatchReceiver):
    """ZMQ-based receiver for AdvantageTrainingBatch (self-contained).

    Phase 10 cluster e2e revealed that inheriting from
    ZMQTrainingBatchReceiver dragged in get_multi_run_manager() at
    __init__ time, a hard dependency on the LLM trainer multi-run
    infrastructure that the AdvTrainer never initializes. The AdvTrainer
    is single-run by design, so this class skips the entire multi-run
    routing layer:

      - PULL socket bind (same as parent pattern)
      - Drain incoming msgpack-encoded batches into a step-keyed dict
      - Return all pending batches in step order on each receive() call

    Matches FileSystemAdvantageTrainingBatchReceiver semantics.
    """

    _batch_type = AdvantageTrainingBatch

    def __init__(self, transport: ZMQTransportConfig) -> None:
        super().__init__()
        # Lazy zmq import: parent transport.zmq does the same so the module
        # stays importable in CPU-only test environments without pyzmq.
        import zmq

        self._zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, transport.hwm)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Single-run: flat step-keyed pending buffer (no per-run-id routing).
        self._pending_by_step: dict[int, AdvantageTrainingBatch] = {}

        self.logger.info(
            f"ZMQ advantage training batch receiver initialized: "
            f"endpoint=tcp://{transport.host}:{transport.port} hwm={transport.hwm}"
        )

    def can_receive(self) -> bool:
        if self._pending_by_step:
            return True
        events = dict(self.poller.poll(timeout=0))
        return self.socket in events

    def _drain_into_pending(self) -> None:
        while True:
            try:
                _sender_id, payload = self.socket.recv_multipart(
                    flags=self._zmq.NOBLOCK, copy=False
                )
            except self._zmq.Again:
                break
            try:
                batch: AdvantageTrainingBatch = self.decoder.decode(payload)
            except Exception as e:
                self.logger.error(f"Error decoding advantage batch: {e}")
                continue
            self._pending_by_step[batch.step] = batch

    def receive(self) -> list[AdvantageTrainingBatch]:  # type: ignore[override]
        self._drain_into_pending()
        if not self._pending_by_step:
            return []
        steps = sorted(self._pending_by_step.keys())
        return [self._pending_by_step.pop(s) for s in steps]

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.info("ZMQ advantage training batch receiver closed")


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
    input_dir: Path | None = None,
) -> FileSystemAdvantageTrainingBatchReceiver | ZMQAdvantageTrainingBatchReceiver:
    """Construct the receiver matching `transport`.

    For filesystem transport, `input_dir` is required (the directory the
    orchestrator writes batches into). For ZMQ transport, `input_dir` is
    ignored.
    """
    if transport.type == "filesystem":
        if input_dir is None:
            raise ValueError(
                "FileSystem advantage transport requires input_dir; the "
                "launcher should wire it to "
                "<orchestrator.output_dir>/advantage_trainer_transport."
            )
        return FileSystemAdvantageTrainingBatchReceiver(input_dir)
    elif transport.type == "zmq":
        return ZMQAdvantageTrainingBatchReceiver(transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")
