"""Held-out contrast evaluation for the branched warm-start (plan branched-dataset
Phase 3 / gate G-A).

A *root* of the branched dataset is a context o_t with several labelled actions:
the trunk's own action (return G) and each branch's forced action (return g).
For every unordered pair of distinct actions at the same root whose mean returns
differ, the warmed Q+ should order them as the returns do. This module
enumerates those pairs from BranchRecords (pure), scores them with any
q_fn(o_ids, action) -> float (pure), and, as a CLI, does so with the trained
value_state.pt and writes gate/contrast.json (branch-schema.md §5).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Iterable

import torch
from pydantic import Field

from prime_rl.advantage_server.action_advantage import tokenize_action
from prime_rl.advantage_trainer.ckpt import load_warm_start
from prime_rl.configs.shared import LogConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone, ValueNetworkConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.dataset_io import BranchRecord, read_branch_records


class ContrastEvalConfig(BaseConfig):
    """Score held-out root pairs with a trained value_state.pt (gate G-A)."""

    value_state: Annotated[Path, Field(description="value_state.pt from warmstart-train.")]
    model: Annotated[
        ValueNetworkConfig,
        Field(description="Value backbone config (same base model + LoRA as training)."),
    ]
    dataset: Annotated[Path, Field(description="The branched dataset (BranchRecords).")]
    holdout_only: Annotated[
        bool,
        Field(description="Use only records flagged holdout (or listed in holdout_ids_path)."),
    ] = True
    holdout_ids_path: Annotated[
        Path | None,
        Field(description="JSON list of game ids treated as held out (in addition to the flag)."),
    ] = None
    out: Annotated[Path, Field(description="Output JSON path.")] = Path("gate/contrast.json")
    min_p_accept: Annotated[
        float | None,
        Field(ge=0.0, le=1.0, description="Ignore branches accepted below this p (BranchRecord.p_accept; D-17)."),
    ] = None
    max_pairs: Annotated[
        int | None,
        Field(description="Score at most this many pairs (deterministic order; None = all)."),
    ] = None
    log: Annotated[LogConfig, Field(description="Logging configuration.")] = LogConfig()


@dataclass
class RootMember:
    """One labelled action at a root."""

    action: str
    g: float
    o_ids: list[int]
    provenance: str  # "trunk" | "branch"
    block_source: str | None = None
    action_source: str | None = None
    branch_id: str | None = None
    p_accept: float | None = None


@dataclass
class Pair:
    game_id: str
    trunk_id: str
    root_turn: int
    a_i: str
    a_j: str
    g_i: float
    g_j: float
    o_ids: list[int]
    meta_i: dict = field(default_factory=dict)
    meta_j: dict = field(default_factory=dict)


def _episode_return(record: BranchRecord) -> float:
    """gamma = 1, terminal-only reward, as build_warmstart_samples: a truncated
    episode has no terminal reward."""
    return float(record.reward) if not record.is_truncated else 0.0


def _joint_dp(record: BranchRecord, joint_idx: int):
    """(sample, decision_point) at a joint decision-point index, or None."""
    j = 0
    for s in record.samples:
        for dp in s.decision_points or []:
            if j == joint_idx:
                return s, dp
            j += 1
    return None


def _member_at(record: BranchRecord, joint_idx: int, provenance: str) -> RootMember | None:
    found = _joint_dp(record, joint_idx)
    if found is None:
        return None
    s, dp = found
    o_ids = list(s.prompt_ids) + list(s.completion_ids[: dp.response_start])
    return RootMember(
        action=dp.admissible_actions[dp.executed_action_idx],
        g=_episode_return(record),
        o_ids=o_ids,
        provenance=provenance,
        block_source=record.block_source,
        action_source=record.action_source,
        branch_id=record.branch_id,
        p_accept=record.p_accept if provenance == "branch" else None,
    )


P_BANDS = ((0.5, "p>=0.5"), (0.1, "0.1<=p<0.5"), (0.01, "0.01<=p<0.1"))


def p_band(p: float) -> str:
    """Acceptance band of a forced root (BranchRecord.p_accept; D-17)."""
    for lo, name in P_BANDS:
        if p >= lo:
            return name
    return "p<0.01"


def _meta(members: list[RootMember]) -> dict:
    """Provenance tags of the member(s) behind one action at a root."""
    return {
        "provenance": sorted({m.provenance for m in members}),
        "block_source": sorted({m.block_source for m in members if m.block_source}),
        "action_source": sorted({m.action_source for m in members if m.action_source}),
        "p_band": sorted({p_band(m.p_accept) for m in members if m.p_accept is not None}),
        "n_episodes": len(members),
    }


def contrast_pairs(
    records: Iterable[BranchRecord],
    *,
    holdout_only: bool = True,
    holdout_ids: set[str] | None = None,
    min_p_accept: float | None = None,
) -> list[Pair]:
    """Enumerate root pairs: for each (trunk_id, root_turn) with at least one
    branch, group members by action (trunk's own action included), average g per
    action, and emit every unordered pair of distinct actions with different mean
    returns. Deterministic order: by (game_id, trunk_id, root_turn, a_i, a_j)."""
    holdout_ids = set(holdout_ids or ())
    recs = list(records)
    trunks: dict[str, BranchRecord] = {}
    roots: dict[tuple[str, int], list[RootMember]] = defaultdict(list)
    root_game: dict[tuple[str, int], str] = {}
    for r in recs:
        if holdout_only and not (r.holdout or r.game_id in holdout_ids):
            continue
        if not r.root_forced:
            trunks[r.trunk_id] = r
            continue
        if r.root_dp is None or r.root_turn is None:
            continue
        if min_p_accept is not None and r.p_accept is not None and r.p_accept < min_p_accept:
            continue
        m = _member_at(r, r.root_dp, "branch")
        if m is None:
            continue
        key = (r.trunk_id, int(r.root_turn))
        roots[key].append(m)
        root_game[key] = r.game_id
    for (trunk_id, turn), members in roots.items():
        trunk = trunks.get(trunk_id)
        if trunk is None:
            continue
        m = _member_at(trunk, turn, "trunk")
        if m is not None:
            members.append(m)

    pairs: list[Pair] = []
    for key in sorted(roots):
        trunk_id, turn = key
        by_action: dict[str, list[RootMember]] = defaultdict(list)
        for m in roots[key]:
            by_action[m.action].append(m)
        actions = sorted(by_action)
        for i_idx in range(len(actions)):
            for j_idx in range(i_idx + 1, len(actions)):
                a_i, a_j = actions[i_idx], actions[j_idx]
                g_i = sum(m.g for m in by_action[a_i]) / len(by_action[a_i])
                g_j = sum(m.g for m in by_action[a_j]) / len(by_action[a_j])
                if g_i == g_j:
                    continue
                pairs.append(
                    Pair(
                        game_id=root_game[key],
                        trunk_id=trunk_id,
                        root_turn=turn,
                        a_i=a_i,
                        a_j=a_j,
                        g_i=g_i,
                        g_j=g_j,
                        o_ids=by_action[a_i][0].o_ids,
                        meta_i=_meta(by_action[a_i]),
                        meta_j=_meta(by_action[a_j]),
                    )
                )
    return pairs


def _split_keys(pair: Pair, field_name: str) -> set[str]:
    """Tags a pair contributes to for one split (union over its two actions;
    a trunk member contributes 'trunk' to the provenance split only)."""
    keys = set(pair.meta_i.get(field_name, [])) | set(pair.meta_j.get(field_name, []))
    return keys


def score_pairs(
    pairs: list[Pair],
    q_fn: Callable[[list[int], str], float],
    *,
    max_pairs: int | None = None,
) -> dict:
    """Score each pair with q_fn(o_ids, action); correct iff Q+ strictly orders
    the two actions as their mean returns do (ties are wrong). Returns the
    contrast.json payload."""
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    cache: dict[tuple[str, int, str], float] = {}

    def q(pair: Pair, action: str) -> float:
        key = (pair.trunk_id, pair.root_turn, action)
        if key not in cache:
            cache[key] = float(q_fn(pair.o_ids, action))
        return cache[key]

    rows = []
    splits: dict[str, dict[str, list[bool]]] = {
        "provenance": defaultdict(list),
        "block_source": defaultdict(list),
        "action_source": defaultdict(list),
        "p_band": defaultdict(list),
    }
    kinds: dict[str, list[bool]] = defaultdict(list)
    for p in pairs:
        q_i, q_j = q(p, p.a_i), q(p, p.a_j)
        # Correct iff Q+ strictly orders the pair as the returns do; a Q+ tie is
        # wrong (a flat Q+ must score 0, not 1).
        correct = (q_i - q_j) * (p.g_i - p.g_j) > 0.0
        rows.append(
            {
                "game_id": p.game_id,
                "trunk_id": p.trunk_id,
                "root_turn": p.root_turn,
                "a_i": p.a_i,
                "a_j": p.a_j,
                "g_i": p.g_i,
                "g_j": p.g_j,
                "q_i": q_i,
                "q_j": q_j,
                "correct": bool(correct),
                "meta_i": p.meta_i,
                "meta_j": p.meta_j,
            }
        )
        for name in splits:
            for k in _split_keys(p, name):
                splits[name][k].append(bool(correct))
        has_trunk = "trunk" in p.meta_i.get("provenance", []) or "trunk" in p.meta_j.get(
            "provenance", []
        )
        kinds["trunk-branch" if has_trunk else "branch-branch"].append(bool(correct))

    def agg(flags: list[bool]) -> dict:
        n = len(flags)
        return {"n": n, "accuracy": (sum(flags) / n) if n else None}

    all_flags = [r["correct"] for r in rows]
    return {
        "schema": 1,
        "n_pairs": len(rows),
        "n_roots": len({(r["trunk_id"], r["root_turn"]) for r in rows}),
        "accuracy": (sum(all_flags) / len(all_flags)) if all_flags else None,
        "by_pair_kind": {k: agg(v) for k, v in sorted(kinds.items())},
        "by_provenance": {k: agg(v) for k, v in sorted(splits["provenance"].items())},
        "by_block_source": {k: agg(v) for k, v in sorted(splits["block_source"].items())},
        "by_action_source": {k: agg(v) for k, v in sorted(splits["action_source"].items())},
        "by_p_band": {k: agg(v) for k, v in sorted(splits["p_band"].items())},
        "pairs": rows,
    }


def _load_holdout_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with open(path) as f:
        return {str(i) for i in json.load(f)}


def build_q_fn(backbone: ValueNetworkBackbone, tokenizer, device) -> Callable[[list[int], str], float]:
    """Q+(o, a) = forward_q_plus_action([o], [tokenize_action(a)]), eval mode, no grad."""
    backbone.eval()

    @torch.no_grad()
    def q_fn(o_ids: list[int], action: str) -> float:
        o = torch.tensor([list(o_ids)], dtype=torch.long, device=device)
        a = torch.tensor([tokenize_action(tokenizer, action)], dtype=torch.long, device=device)
        return float(backbone.forward_q_plus_action(o, a)[0].float().item())

    return q_fn


def contrast_eval(config: ContrastEvalConfig) -> dict:
    logger = setup_logger(config.log.level)
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.bfloat16
    else:
        device, dtype = torch.device("cpu"), torch.float32
    logger.info(f"Loading value backbone from {config.model.base_model_name} ({device}, {dtype})")
    base = AutoModel.from_pretrained(config.model.base_model_name, dtype=dtype)
    backbone = ValueNetworkBackbone(
        base, lora_config=config.model.lora, polyak_tau=config.model.polyak_tau
    ).to(device=device, dtype=dtype)
    load_warm_start(backbone, config.value_state)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)

    pairs = contrast_pairs(
        read_branch_records(config.dataset),
        holdout_only=config.holdout_only,
        holdout_ids=_load_holdout_ids(config.holdout_ids_path),
        min_p_accept=config.min_p_accept,
    )
    if not pairs:
        raise ValueError(
            f"No contrast pairs in {config.dataset} (holdout_only={config.holdout_only}). "
            "Check the collect's holdout flags or pass holdout_ids_path."
        )
    logger.info(f"Scoring {len(pairs)} root pairs" + (f" (max_pairs={config.max_pairs})" if config.max_pairs else ""))
    payload = score_pairs(pairs, build_q_fn(backbone, tokenizer, device), max_pairs=config.max_pairs)
    payload["value_state"] = str(config.value_state)
    payload["dataset"] = str(config.dataset)
    Path(config.out).parent.mkdir(parents=True, exist_ok=True)
    with open(config.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.success(
        f"contrast accuracy={payload['accuracy']:.3f} over {payload['n_pairs']} pairs "
        f"({payload['n_roots']} roots) -> {config.out}"
    )
    return payload


def main() -> None:
    """Entry point: `python -m prime_rl.value_warmstart.contrast_eval @ contrast.toml`."""
    contrast_eval(cli(ContrastEvalConfig))


if __name__ == "__main__":
    main()
