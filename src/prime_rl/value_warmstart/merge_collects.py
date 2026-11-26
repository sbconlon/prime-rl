"""Merge several branch-collect folders into one (plan bulletproof-warm-start: the
gold test set is collected by several jobs, one per source or slice of games).

Only finished games (a done marker) are taken; records of other games are left
out and counted. A game found in two inputs is an error. The output folder holds
records.msgpack, nodes/, done/, collect_stats.json (the inputs' counters added
up) and sources.json (input name -> its game ids), which the push score reads to
split its numbers by source.

    python -m prime_rl.value_warmstart.merge_collects @ merge.toml
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import fields
from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.branch_records import CollectStats
from prime_rl.value_warmstart.dataset_io import _BRANCH_DECODER, open_records, records_exist


class MergeCollectsConfig(BaseConfig):
    """Join branch-collect output folders into one."""

    inputs: Annotated[list[Path], Field(min_length=1, description="Collect output folders to join.")]
    names: Annotated[
        list[str],
        Field(description="One name per input for sources.json (default: the folder names)."),
    ] = []
    out: Annotated[Path, Field(description="Output folder. Must not hold records.msgpack yet.")]
    log: Annotated[LogConfig, Field(description="Logging configuration.")] = LogConfig()


def _whole_frames(path: Path):
    """(header, blob) of every complete frame; a cut-off last frame is skipped."""
    with open_records(path) as f:
        while True:
            header = f.read(4)
            if len(header) < 4:
                return
            (n,) = struct.unpack("<I", header)
            blob = f.read(n)
            if len(blob) < n:
                return
            yield header, blob


def add_stats(stats: list[CollectStats]) -> CollectStats:
    """Counter by counter sum (dict counters key by key)."""
    total = CollectStats()
    for s in stats:
        for f in fields(CollectStats):
            a, b = getattr(total, f.name), getattr(s, f.name)
            if isinstance(a, dict):
                for k, v in b.items():
                    a[k] = a.get(k, 0) + v
            else:
                setattr(total, f.name, a + b)
    return total


def merge_collects(inputs: list[Path], out: Path, names: list[str] | None = None) -> dict:
    inputs = [Path(p) for p in inputs]
    names = list(names) if names else [p.name for p in inputs]
    if len(names) != len(inputs):
        raise ValueError(f"{len(names)} name(s) for {len(inputs)} input(s)")
    if len(set(names)) != len(names):
        raise ValueError(f"input names must differ: {names}")
    out = Path(out)
    records_out = out / "records.msgpack"
    if records_out.exists():
        raise FileExistsError(f"{records_out} exists: merge into a new folder")

    done: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for name, src in zip(names, inputs):
        if not records_exist(src / "records.msgpack"):
            raise FileNotFoundError(f"no records.msgpack in {src}")
        ids = sorted(p.stem for p in (src / "done").glob("*.json")) if (src / "done").is_dir() else []
        for g in ids:
            if g in owner:
                raise ValueError(f"game {g} is in both {owner[g]!r} and {name!r}")
            owner[g] = name
        done[name] = ids

    (out / "done").mkdir(parents=True, exist_ok=True)
    report: dict = {"sources": {}}
    stats: list[CollectStats] = []
    with open(records_out, "wb") as dst:
        for name, src in zip(names, inputs):
            finished = set(done[name])
            kept = left_out = 0
            for header, blob in _whole_frames(src / "records.msgpack"):
                if _BRANCH_DECODER.decode(blob).game_id in finished:
                    dst.write(header)
                    dst.write(blob)
                    kept += 1
                else:
                    left_out += 1
            for g in done[name]:
                shutil.copy2(src / "done" / f"{g}.json", out / "done" / f"{g}.json")
                if (src / "nodes" / g).is_dir():
                    shutil.copytree(src / "nodes" / g, out / "nodes" / g)
            if (src / "collect_stats.json").exists():
                stats.append(CollectStats.load(src / "collect_stats.json"))
            report["sources"][name] = {
                "dir": str(src),
                "games": done[name],
                "records": kept,
                "records_left_out": left_out,
            }
    add_stats(stats).save(out / "collect_stats.json")
    report["games"] = len(owner)
    report["records"] = sum(s["records"] for s in report["sources"].values())
    (out / "sources.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def read_sources(gold_dir: Path) -> dict[str, str]:
    """game id -> source name, from a merged folder's sources.json ({} when absent)."""
    path = Path(gold_dir) / "sources.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(g): name for name, s in raw.get("sources", {}).items() for g in s.get("games", [])}


def main() -> None:
    config = cli(MergeCollectsConfig)
    logger = setup_logger(config.log.level)
    report = merge_collects(config.inputs, config.out, config.names or None)
    for name, s in report["sources"].items():
        logger.info(
            f"{name}: {len(s['games'])} game(s), {s['records']} record(s) "
            f"({s['records_left_out']} of unfinished games left out)"
        )
    logger.success(f"merged {report['games']} game(s), {report['records']} record(s) -> {config.out}")


if __name__ == "__main__":
    main()
