"""Framed-msgpack IO for the warm-start dataset: a stream of length-prefixed
WarmStartRecords. Each record is one rollout's interleaved TrainingSamples plus
its terminal reward and truncation flag; the MC targets are computed at load time
(build_warmstart_samples), so gamma/n_step experiments need no regeneration.
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


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(WarmStartRecord)


class RecordWriter:
    """Streaming writer of length-prefixed msgpack frames."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._f = None

    def __enter__(self) -> "RecordWriter":
        self._f = open(self._path, "wb")
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()

    def write(self, record: WarmStartRecord) -> None:
        blob = _ENCODER.encode(record)
        self._f.write(struct.pack("<I", len(blob)))
        self._f.write(blob)


def read_records(path: Path) -> Iterator[WarmStartRecord]:
    """Yield each WarmStartRecord from a framed dataset file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                return
            (n,) = struct.unpack("<I", header)
            yield _DECODER.decode(f.read(n))
