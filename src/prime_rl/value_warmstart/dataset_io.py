"""Framed-msgpack IO for the warm-start datasets: a stream of length-prefixed
records. Each record is one episode's interleaved TrainingSamples plus its
terminal reward and truncation flag; the MC targets are computed at load time
(build_warmstart_samples), so gamma/n_step experiments need no regeneration.

Two record types share the frame format:

* WarmStartRecord -- the single-action (Job-A) dataset of warmstart-collect.
* BranchRecord    -- the branched (Job-B) dataset of branch-collect: a superset
  whose first three fields are WarmStartRecord's, plus where the labels start,
  which decision point is the forced root, and provenance. See
  plan/branched-dataset/branch-schema.md §3.

Readers: read_records yields WarmStartRecords from either file kind (a
BranchRecord's trailing fields are ignored); read_branch_records yields
BranchRecords and promotes a legacy WarmStartRecord to a trunk-like
BranchRecord (label_from_dp=0, root_forced=False).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import msgspec

from prime_rl.transport.types import TrainingSample


class WarmStartRecord(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One rollout: its interleaved TrainingSamples (with decision_points), the
    episode's terminal reward, and whether the episode was truncated."""

    samples: list[TrainingSample]
    reward: float
    is_truncated: bool


class BranchRecord(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """One episode of the branched dataset (trunk or branch). Index fields are in
    the JOINT decision-point sequence of `samples` (the order build_warmstart_samples
    flattens), so a truncation split never shifts them."""

    samples: list[TrainingSample]
    reward: float
    is_truncated: bool
    # First decision point that receives targets (trunk: 0; branch: the root).
    label_from_dp: int = 0
    # The forced decision point (branch) -- its target gets v_weight = 0. None: trunk.
    root_dp: int | None = None
    root_forced: bool = False
    game_id: str = ""
    trunk_id: str = ""
    branch_id: str | None = None
    root_turn: int | None = None
    action_source: str | None = None  # "own" | "tail"
    block_source: str | None = None  # "sampled" | "elicited"
    p_accept: float | None = None
    allocation_stage: int | None = None
    cont_len: int = 0
    holdout: bool = False


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(WarmStartRecord)
_BRANCH_DECODER = msgspec.msgpack.Decoder(BranchRecord)


class RecordWriter:
    """Streaming writer of length-prefixed msgpack frames (either record type)."""

    def __init__(self, path: Path, append: bool = False):
        self._path = Path(path)
        self._append = append
        self._f = None

    def __enter__(self) -> "RecordWriter":
        self._f = open(self._path, "ab" if self._append else "wb")
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()

    def write(self, record: WarmStartRecord | BranchRecord) -> None:
        blob = _ENCODER.encode(record)
        self._f.write(struct.pack("<I", len(blob)))
        self._f.write(blob)
        self._f.flush()


def _frames(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                return
            (n,) = struct.unpack("<I", header)
            yield f.read(n)


def read_records(path: Path) -> Iterator[WarmStartRecord]:
    """Yield each record as a WarmStartRecord (trailing BranchRecord fields ignored)."""
    for blob in _frames(path):
        yield _DECODER.decode(blob)


def read_branch_records(path: Path) -> Iterator[BranchRecord]:
    """Yield each record as a BranchRecord; a legacy 3-field WarmStartRecord frame
    decodes to a trunk-like BranchRecord (defaults for the missing fields)."""
    for blob in _frames(path):
        yield _BRANCH_DECODER.decode(blob)
