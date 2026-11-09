"""Branch-collect helpers: seeds, trajectory splicing, BranchRecords, node JSON,
and the running statistics file (plan branched-dataset, branch-schema.md §1/§3,
master doc D-13/D-14).

Everything here is pure over plain dicts / wire types so it is unit-testable
without an env, a client, or an engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prime_rl.transport.types import TrainingSample
from prime_rl.value_warmstart.dataset_io import BranchRecord

# One verb-class table for the collect stats, the probe gate metrics, and the
# thesis gate_check (Phase 4): prime_rl.utils.alfworld_actions.
from prime_rl.utils.alfworld_actions import VERB_CLASSES, verb_class  # noqa: E402,F401


def game_id_from_file(game_file: str) -> str:
    """<task dir>__<trial dir>, filesystem-safe and unique within a split."""
    p = Path(game_file)
    return f"{p.parent.parent.name}__{p.parent.name}"


def trunk_id_for(game_id: str) -> str:
    """Short, unique trunk id: nodes/<game_id>/<trunk_id>/ must stay under Windows
    MAX_PATH when the run is pulled to the laptop (game ids run ~90 chars)."""
    return hashlib.sha1(game_id.encode("utf-8")).hexdigest()[:10] + "-trunk"


def _role(m: Any) -> str | None:
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _content(m: Any) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return c if isinstance(c, str) else ""


def step_executed_action(step: dict) -> str:
    return str(step["extras"]["executed_action"])


def step_admissible(step: dict) -> list[str]:
    return [str(a) for a in step["extras"]["admissible_actions"]]


def make_seed(
    trunk_out: dict,
    t: int,
    action: str,
    forced_completion: str,
    trunk_id: str,
    branch_id: str,
) -> dict:
    """The small, JSON-like branch seed the env's seeded setup_state consumes.

    The branch example's prompt is the trunk's WINDOWED prompt at the root (what
    the trunk itself would have continued from after eviction), so the seed's
    completion_prefix is empty: the env builds prompt + [forced turn, new obs].
    """
    steps = trunk_out["trajectory"]
    if not (0 <= t < len(steps)):
        raise ValueError(f"root turn {t} outside trunk of {len(steps)} steps")
    return {
        "trunk_id": trunk_id,
        "branch_id": branch_id,
        "actions_prefix": [step_executed_action(s) for s in steps[:t]],
        "expected_obs": _content(steps[t]["prompt"][-1]),
        "forced_action": action,
        "forced_completion": forced_completion,
        "completion_prefix": [],
        "turn": t,
    }


def branch_example(trunk_example: dict, trunk_out: dict, seed: dict) -> dict:
    """The rollout input for a branch: the trunk's windowed prompt at the root
    (trunk step t's prompt messages) plus the seed in info."""
    ex = dict(trunk_example)
    ex["prompt"] = list(trunk_out["trajectory"][seed["turn"]]["prompt"])
    info = dict(ex.get("info") or {})
    info["branch_seed"] = seed
    ex["info"] = info
    return ex


def splice_output(branch_out: dict, prefix_steps: list, forced_step: dict) -> dict:
    """The branch RolloutOutput with its trajectory = trunk prefix + forced step +
    continuation, so interleave_rollout yields decision points for every turn."""
    out = dict(branch_out)
    out["trajectory"] = [copy.deepcopy(s) for s in prefix_steps] + [forced_step] + list(
        branch_out.get("trajectory") or []
    )
    return out


def joint_dp_count(samples: list[TrainingSample]) -> int:
    return sum(len(s.decision_points or []) for s in samples)


def trunk_record(
    samples: list[TrainingSample], out: dict, *, game_id: str, trunk_id: str, holdout: bool
) -> BranchRecord:
    return BranchRecord(
        samples=samples,
        reward=float(out["reward"]),
        is_truncated=bool(out.get("is_truncated", False)),
        label_from_dp=0,
        root_dp=None,
        root_forced=False,
        game_id=game_id,
        trunk_id=trunk_id,
        branch_id=None,
        root_turn=None,
        cont_len=max(0, joint_dp_count(samples) - 1),
        holdout=holdout,
    )


def branch_record(
    samples: list[TrainingSample],
    out: dict,
    *,
    game_id: str,
    trunk_id: str,
    branch_id: str,
    root_turn: int,
    action_source: str,
    block_source: str,
    p_accept: float | None,
    allocation_stage: int | None,
    holdout: bool,
) -> BranchRecord:
    total = joint_dp_count(samples)
    if total < root_turn + 1:
        raise ValueError(f"branch has {total} decision points but the root is at {root_turn}")
    return BranchRecord(
        samples=samples,
        reward=float(out["reward"]),
        is_truncated=bool(out.get("is_truncated", False)),
        label_from_dp=root_turn,
        root_dp=root_turn,
        root_forced=True,
        game_id=game_id,
        trunk_id=trunk_id,
        branch_id=branch_id,
        root_turn=root_turn,
        action_source=action_source,
        block_source=block_source,
        p_accept=p_accept,
        allocation_stage=allocation_stage,
        cont_len=total - root_turn - 1,
        holdout=holdout,
    )


# ---------------------------------------------------------------------------
# Node JSON (branch-schema.md §1)
# ---------------------------------------------------------------------------


def node_json(
    *,
    game_id: str,
    trunk_id: str,
    turn: int,
    admissible: list[str],
    a_taken: str,
    trunk_return: float,
    node: Any,
    chosen: list[dict],
    allocation: str,
    stage: int | None,
    pivotal: bool | None,
    replay_ok: bool,
    timing: dict,
) -> dict:
    blocks = []
    for i, b in enumerate(node.blocks):
        blocks.append(
            {
                "reasoning": getattr(b, "reasoning", ""),
                "tag_action": getattr(b, "tag_action", ""),
                "row": [float(x) for x in node.rows[i]],
                "mass": float(node.masses[i]) if i < len(node.masses) else None,
            }
        )
    return {
        "schema": 1,
        "game_id": game_id,
        "trunk_id": trunk_id,
        "turn": turn,
        "admissible": list(admissible),
        "a_taken": a_taken,
        "trunk_return": float(trunk_return),
        "blocks": blocks,
        "lead": list(node.lead),
        "own": list(node.own),
        "chosen": chosen,
        "allocation": allocation,
        "stage": stage,
        "pivotal": pivotal,
        "replay_ok": replay_ok,
        "timing": timing,
    }


def write_node_json(nodes_dir: Path, game_id: str, trunk_id: str, turn: int, payload: dict) -> Path:
    d = Path(nodes_dir) / game_id / trunk_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"t{turn}.json"
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Running statistics (collect_stats.json; the gate's G-E input)
# ---------------------------------------------------------------------------


@dataclass
class CollectStats:
    games_done: int = 0
    games_skipped_resume: int = 0
    trunks: int = 0
    trunks_dropped: int = 0
    trunk_positives: int = 0
    roots: int = 0
    roots_with_contrast: int = 0
    branches_run: int = 0
    branches_dropped: int = 0
    branch_positives: int = 0
    branches_skipped: dict[str, int] = field(default_factory=dict)
    accept_by_block_source: dict[str, int] = field(default_factory=dict)
    attempts_by_action_source: dict[str, int] = field(default_factory=dict)
    accept_by_action_source: dict[str, int] = field(default_factory=dict)
    accept_weak_by_action_source: dict[str, int] = field(default_factory=dict)
    root_labels_by_verb_class: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    admissible_mass_at_roots_sum: float = 0.0
    admissible_mass_at_roots_n: int = 0
    stage1_roots: int = 0
    stage2_roots: int = 0
    replay_failures: int = 0
    generations_est: int = 0

    @staticmethod
    def _bump(d: dict[str, int], key: str, n: int = 1) -> None:
        d[key] = d.get(key, 0) + n

    def to_json(self) -> dict:
        d = asdict(self)
        n = self.admissible_mass_at_roots_n
        d["admissible_mass_at_roots_mean"] = (self.admissible_mass_at_roots_sum / n) if n else None
        d["roots_with_contrast_frac"] = (self.roots_with_contrast / self.roots) if self.roots else None
        return d

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CollectStats":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in names})
