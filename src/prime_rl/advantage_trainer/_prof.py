"""Per-stage profiling for the Advantage Trainer training loop.

Lightweight: a context manager that times a code block via `time.perf_counter`,
optionally syncs CUDA before the stop measurement (so GPU work isn\'t
counted asynchronously), and emits a structured log line:

    PROF rid=<step> stage=<name> duration_ms=<X.X> [extras...]

A correlation id is propagated via a contextvar; the AdvTrainer loop sets
it to `str(step)` once per training step, and every `prof()` block emits
with that id. The analyzer (`scripts/canary/parse_prof_log.py`) groups by
stage and prints mean/p50/p95/max/total per step.

Activation: profiling is on iff env var ``PRIME_RL_ADVTRAINER_PROF`` is set
to a truthy value (e.g. ``1``). Off by default so production runs pay zero
cost.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch

_PROF_LOGGER = logging.getLogger("prime_rl.advantage_trainer.prof")
_REQUEST_ID: ContextVar[str] = ContextVar("_REQUEST_ID", default="-")


def is_profiling_enabled() -> bool:
    val = os.environ.get("PRIME_RL_ADVTRAINER_PROF", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def new_request_id() -> str:
    """Return a short id for correlating PROF lines from one training step."""
    return uuid.uuid4().hex[:8]


def set_request_id(rid: str) -> None:
    _REQUEST_ID.set(rid)


def get_request_id() -> str:
    return _REQUEST_ID.get()


@contextmanager
def prof(stage: str, *, sync_cuda: bool = False, **extras: object) -> Iterator[None]:
    """Time the wrapped block and emit a PROF log line.

    Args:
        stage: hierarchical stage name, e.g. "advtrainer.forward".
        sync_cuda: if True, call torch.cuda.synchronize() before stopping the
            timer. Required for any GPU-bound stage so we measure actual wall
            time, not just dispatch.
        **extras: free-form key=value pairs appended to the log line
            (e.g. K=8 S=2048 epoch=3). All values are str-coerced.
    """
    if not is_profiling_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rid = _REQUEST_ID.get()
        extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
        suffix = f" {extras_str}" if extras_str else ""
        _PROF_LOGGER.info(
            f"PROF rid={rid} stage={stage} duration_ms={elapsed_ms:.3f}{suffix}"
        )
