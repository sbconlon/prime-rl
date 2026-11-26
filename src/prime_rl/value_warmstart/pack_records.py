"""Pack a finished collect's records.msgpack with xz, losslessly (plan
bulletproof-warm-start: every episode in a records file repeats its state's whole
conversation, so the file shrinks about a hundred times, and the home quota cannot
hold the plain files of a large collect).

The packed copy is records.msgpack.xz next to the plain file. Every reader of a
records file falls back to it when the plain file is gone (dataset_io.open_records),
so the scorer, the trainer and merge_collects need no change of config.

The plain file is removed only with replace = true, and only after the packed copy
was read back and found identical (same length, same sha256) and the plain file
was not written to meanwhile. Without replace both files stay, and a later call
with replace = true checks the packed copy again before removing the plain one.
Pack finished collects only. To unpack by hand: xz -dk records.msgpack.xz

    python -m prime_rl.value_warmstart.pack_records @ pack.toml
"""

from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.dataset_io import packed_path

_CHUNK = 1 << 22


class PackRecordsConfig(BaseConfig):
    """Pack the records of finished collect folders."""

    dirs: Annotated[list[Path], Field(min_length=1, description="Collect folders (each holds records.msgpack).")]
    replace: Annotated[bool, Field(description="Remove the plain file once the packed copy is verified.")] = False
    preset: Annotated[int, Field(ge=0, le=9, description="xz preset.")] = 6
    log: Annotated[LogConfig, Field(description="Logging configuration.")] = LogConfig()


def _stamp(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def _digest(f, sink=None) -> tuple[int, str]:
    """(bytes, sha256) of a stream; every chunk is also written to ``sink``."""
    h = hashlib.sha256()
    n = 0
    while chunk := f.read(_CHUNK):
        h.update(chunk)
        n += len(chunk)
        if sink is not None:
            sink.write(chunk)
    return n, h.hexdigest()


def _pack_to(src: Path, dst: Path, preset: int) -> tuple[int, str]:
    with open(src, "rb") as f, lzma.open(dst, "wb", preset=preset) as z:
        return _digest(f, z)


def pack_records(path: Path, replace: bool = False, preset: int = 6) -> dict:
    """Pack one records file. Returns {"records", "status", "plain_bytes",
    "packed_bytes", "sha256", "replaced"}; status is "packed", "verified" (a packed
    copy was already there and matches) or "already packed" (no plain file left)."""
    path = Path(path)
    packed = packed_path(path)
    if not path.exists():
        if not packed.exists():
            raise FileNotFoundError(f"no {path}")
        return {
            "records": str(path),
            "status": "already packed",
            "plain_bytes": None,
            "packed_bytes": packed.stat().st_size,
            "sha256": None,
            "replaced": False,
        }
    before = _stamp(path)
    tmp = Path(str(packed) + ".tmp")
    fresh = not packed.exists()
    if fresh:
        plain = _pack_to(path, tmp, preset)
        candidate = tmp
    else:
        with open(path, "rb") as f:
            plain = _digest(f)
        candidate = packed
    with lzma.open(candidate, "rb") as z:
        back = _digest(z)
    if back != plain:
        if fresh:
            tmp.unlink()
        raise ValueError(f"the packed copy of {path} does not read back identical: nothing removed")
    if _stamp(path) != before:
        if fresh:
            tmp.unlink()
        raise RuntimeError(f"{path} changed while it was packed (a collect still writing?): nothing removed")
    if fresh:
        tmp.replace(packed)
    n, digest = plain
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    if replace:
        path.unlink()
    return {
        "records": str(path),
        "status": "packed" if fresh else "verified",
        "plain_bytes": n,
        "packed_bytes": packed.stat().st_size,
        "sha256": digest,
        "replaced": replace,
    }


def main() -> None:
    config = cli(PackRecordsConfig)
    logger = setup_logger(config.log.level)
    for d in config.dirs:
        r = pack_records(Path(d) / "records.msgpack", replace=config.replace, preset=config.preset)
        if r["plain_bytes"] is None:
            logger.info(f"{d}: already packed ({r['packed_bytes'] / 1e6:.1f} MB)")
            continue
        logger.success(
            f"{d}: {r['status']}, {r['plain_bytes'] / 1e6:.1f} MB -> {r['packed_bytes'] / 1e6:.1f} MB"
            + (", plain file removed" if r["replaced"] else ", plain file kept")
        )


if __name__ == "__main__":
    main()
