"""Branched dataset Phase 2: allocation, seeds, splicing, records, and the driver
wiring (fake node env, scripted rollouts, stubbed interleave and pick_actions).
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.configs.value_warmstart import BranchCollectConfig
from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart import branch_collect as bc
from prime_rl.value_warmstart.allocation import even_roots, pivotal_roots, stratified_roots
from prime_rl.value_warmstart.branch_records import (
    CollectStats,
    branch_example,
    branch_record,
    game_id_from_file,
    make_seed,
    splice_output,
    trunk_record,
    verb_class,
)
from prime_rl.value_warmstart.dataset_io import RecordWriter, read_branch_records

# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------


def test_stratified_roots_tilt_and_no_terminal():
    roots = stratified_roots(T=20, B=6, early_tilt=0.5)
    assert len(roots) == 6
    assert 19 not in roots
    # candidates 0..18, first third = 0..5 -> three early roots
    assert sum(1 for t in roots if t <= 5) == 3
    assert roots == sorted(set(roots))


def test_stratified_roots_small_trunk_returns_all_but_terminal():
    assert stratified_roots(T=4, B=6) == [0, 1, 2]
    assert stratified_roots(T=1, B=6) == []
    assert stratified_roots(T=0, B=6) == []


def test_stratified_roots_full_tilt():
    roots = stratified_roots(T=30, B=4, early_tilt=1.0)
    assert len(roots) == 4 and max(roots) <= 9


def test_even_roots_early_middle_late():
    assert even_roots(T=15, B=3) == [2, 7, 11]
    assert even_roots(T=10, B=3) == [1, 4, 7]
    assert even_roots(T=6, B=3) == [0, 2, 4]
    # never the terminal turn; short trunks give every candidate
    assert even_roots(T=4, B=3) == [0, 1, 2] and even_roots(T=3, B=3) == [0, 1]
    assert even_roots(T=1, B=3) == [] and even_roots(T=9, B=0) == []
    for T in range(2, 40):
        for B in range(1, 8):
            r = even_roots(T, B)
            assert r == sorted(set(r)) and len(r) == min(B, T - 1) and all(0 <= t < T - 1 for t in r)


def test_pivotal_roots_rule():
    stage1 = {2: 1.0, 5: 0.0, 7: None, 9: 1.0, 11: 1.0}
    assert pivotal_roots(stage1, trunk_return=0.0, B2=2) == [2, 9]
    assert pivotal_roots(stage1, trunk_return=1.0, B2=4) == [5]


# ---------------------------------------------------------------------------
# records helpers
# ---------------------------------------------------------------------------


def _step(t: int, action: str, admissible=("go to desk 1", "open drawer 1", "look")):
    return {
        "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"obs{t}"}],
        "completion": [{"role": "assistant", "content": f"<think>r{t}</think><action>{action}</action>"}],
        "tokens": {"prompt_ids": [1], "prompt_mask": [0], "completion_ids": [2], "completion_mask": [1], "completion_logprobs": [0.0]},
        "extras": {"admissible_actions": list(admissible), "executed_action": action},
    }


def _trunk_out(T=4, reward=0.0):
    steps = [_step(t, "go to desk 1") for t in range(T)]
    completion = []
    for t in range(T):
        completion.append(steps[t]["completion"][0])
        completion.append({"role": "user", "content": f"obs{t + 1}"})
    return {
        "example_id": 0, "task": "alfworld", "prompt": steps[0]["prompt"], "completion": completion,
        "reward": reward, "is_truncated": False, "error": None, "trajectory": steps,
        "sampling_args": {"temperature": 0.8}, "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"},
    }


def _sample(n_dps):
    return TrainingSample(
        prompt_ids=[1], prompt_mask=[False], completion_ids=list(range(n_dps)), completion_mask=[True] * n_dps,
        completion_logprobs=[0.0] * n_dps, completion_temperatures=[1.0] * n_dps,
        decision_points=[DecisionPoint(i, i + 1, ["a", "b"], 0) for i in range(n_dps)],
    )


def test_verb_class_and_game_id():
    assert verb_class("go to desk 1") == "go"
    assert verb_class("clean soapbar 1 with sinkbasin 1") == "clean"
    assert verb_class("frobnicate") == "other"
    assert verb_class("move soapbar 1 to sinkbasin 1") == "move"
    assert game_id_from_file("/d/train/pps-x/trial_1/game.tw-pddl") == "pps-x__trial_1"


def test_seed_and_branch_example():
    out = _trunk_out(T=4)
    seed = make_seed(out, 2, "open drawer 1", "<think>x</think><action>open drawer 1</action>", "tr", "br")
    assert seed["actions_prefix"] == ["go to desk 1", "go to desk 1"]
    assert seed["expected_obs"] == "obs2"
    assert seed["turn"] == 2 and seed["forced_action"] == "open drawer 1"
    # the windowed context at the root travels as the example prompt, so the
    # seed's own prefix is empty (Phase 1 seeded setup_state contract)
    assert seed["completion_prefix"] == []
    ex = branch_example({"prompt": [], "info": {"game_file": "g"}, "example_id": 3}, out, seed)
    assert ex["prompt"] == out["trajectory"][2]["prompt"]
    assert ex["info"]["branch_seed"] is seed and ex["info"]["game_file"] == "g"
    with pytest.raises(ValueError):
        make_seed(out, 9, "look", "x", "tr", "br")


def test_splice_and_record_indices():
    out = _trunk_out(T=5)
    forced = _step(2, "open drawer 1")
    bout = {"reward": 1.0, "is_truncated": False, "trajectory": [_step(3, "look"), _step(4, "look")]}
    spliced = splice_output(bout, out["trajectory"][:2], forced)
    assert len(spliced["trajectory"]) == 5
    samples = [_sample(3), _sample(2)]  # a truncation split
    rec = branch_record(
        samples, bout, game_id="g", trunk_id="t", branch_id="b", root_turn=2, action_source="tail",
        block_source="elicited", p_accept=0.6, allocation_stage=None, holdout=False,
    )
    assert (rec.label_from_dp, rec.root_dp, rec.cont_len, rec.root_forced) == (2, 2, 2, True)
    tr = trunk_record([_sample(5)], out, game_id="g", trunk_id="t", holdout=True)
    assert (tr.label_from_dp, tr.root_dp, tr.root_forced, tr.cont_len, tr.holdout) == (0, None, False, 4, True)
    with pytest.raises(ValueError):
        branch_record([_sample(1)], bout, game_id="g", trunk_id="t", branch_id="b", root_turn=2,
                      action_source="own", block_source="sampled", p_accept=0.9, allocation_stage=1, holdout=False)


def test_stats_roundtrip(tmp_path):
    s = CollectStats(roots=4, roots_with_contrast=1)
    CollectStats._bump(s.branches_skipped, "no_congruent_block")
    s.admissible_mass_at_roots_sum, s.admissible_mass_at_roots_n = 1.8, 2
    p = tmp_path / "stats.json"
    s.save(p)
    raw = json.loads(p.read_text())
    assert raw["admissible_mass_at_roots_mean"] == pytest.approx(0.9)
    assert raw["roots_with_contrast_frac"] == pytest.approx(0.25)
    back = CollectStats.load(p)
    assert back.roots == 4 and back.branches_skipped == {"no_congruent_block": 1}


def test_greedy_sampling_args():
    g = bc.greedy_sampling_args({"temperature": 0.8, "max_tokens": 512, "extra_body": {"min_p": 0.1}})
    assert g["temperature"] == 0.0 and g["extra_body"]["min_p"] == 0.0 and g["max_tokens"] == 512


# ---------------------------------------------------------------------------
# driver wiring
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    reasoning: str
    tag_action: str


@dataclass
class _Node:
    blocks: list
    rows: list
    masses: list
    lead: list
    own: list


@dataclass
class _Syn:
    reasoning: str
    source: str
    p_accept: float
    attempts: int
    reject_reasons: list
    block_idx: int | None
    completion_text: str


class _FakeNodeEnv:
    def __init__(self, own, fail_actions=()):
        self.own = own
        self.fail_actions = set(fail_actions)
        self.scored = []

    async def score_node(self, client, model, prompt, admissible, m, block_args, greedy_args):
        self.scored.append(prompt[-1]["content"])
        rows = [[1.0 if a == self.own[0] else 0.0 for a in admissible]] * (m + 1)
        return _Node(blocks=[_Block("r", self.own[0])] * (m + 1), rows=rows, masses=[0.9] * (m + 1),
                     lead=[self.own[0]] * (m + 1), own=list(self.own))

    async def synthesize_block(self, client, model, prompt, admissible, action, node, block_args):
        if action in self.fail_actions:
            # Phase 1 contract: failure is a Synthesis with source "none"
            return _Syn("", "none", 0.0, 3, ["argmax", "tau", "leak"], None, "")
        src = "sampled" if action in node.own else "elicited"
        return _Syn("why", src, 0.8, 0 if src == "sampled" else 1, [], 0 if src == "sampled" else None,
                    f"<think>why</think><action>{action}</action>")

    def build_forced_step(self, trunk_step, syn, action, branch_id, admissible):
        return _step(0, action, admissible)


def _stub_pick(own, admissible, a_taken, k, n_unif, rng):
    own = [a for a in own if a in admissible and a != a_taken]
    chosen = [(a, "own") for a in own[: k - n_unif]]
    tail = [a for a in admissible if a not in own and a != a_taken]
    n_tail = min(k - len(chosen), len(tail))
    chosen += [(a, "tail") for a in rng.sample(tail, n_tail)]
    return chosen


def _cfg(tmp_path, **kw):
    base = dict(
        model_name="sft", env=EnvConfig(id="alfworld-env"), client=ClientConfig(),
        output_path=tmp_path / "records.msgpack", output_dir=tmp_path, roots_per_trunk=2,
        branches_per_root=2, n_uniform=1, node_blocks=2, holdout_games=0, max_concurrent=4,
    )
    base.update(kw)
    return BranchCollectConfig(**base)


class _Writer:
    def __init__(self):
        self.records = []

    def write(self, r):
        self.records.append(r)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))


def _make_ctx(tmp_path, cfg, node_env, n_games=1):
    async def _client():
        return "client"

    return bc._Ctx(
        config=cfg, env="env", node_env=node_env, get_client=_client, model_name="sft",
        sampling_args={"temperature": 0.8, "extra_body": {"min_p": 0.1}}, block_args={}, greedy_args={},
        writer=_Writer(), out_dir=tmp_path, logger=_Logger(), stats=CollectStats(), n_games=n_games,
        sem=asyncio.Semaphore(4),
    )


def _install_fakes(monkeypatch, branch_rewards: dict, drop_branch=None):
    """run_rollout: trunk -> fixed 5-step failure; branch -> reward from the map
    keyed by forced action (None in the map = dropped)."""

    async def fake_run_rollout(env, client, example, model_name, sampling_args, max_retries):
        seed = (example.get("info") or {}).get("branch_seed")
        if seed is None:
            return _trunk_out(T=5, reward=0.0)
        a = seed["forced_action"]
        if drop_branch and a in drop_branch:
            raise RuntimeError("wedged")
        cont = 5 - seed["turn"] - 1
        return {"reward": branch_rewards.get(a, 0.0), "is_truncated": False, "error": None,
                "trajectory": [_step(seed["turn"] + 1 + i, "look") for i in range(cont)],
                "completion": [{"role": "assistant", "content": "x"}], "example_id": 0,
                "sampling_args": {"temperature": 0.8}}

    def fake_interleave(output):
        return [_sample(len(output["trajectory"]))]

    monkeypatch.setattr(bc, "run_rollout", fake_run_rollout)
    monkeypatch.setattr(bc, "interleave_rollout", fake_interleave)
    monkeypatch.setattr(bc, "_pick_actions", _stub_pick)


def test_collect_game_stratified(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"open drawer 1": 1.0, "look": 0.0})
    cfg = _cfg(tmp_path)
    node_env = _FakeNodeEnv(own=["go to desk 1", "open drawer 1"])
    ctx = _make_ctx(tmp_path, cfg, node_env)
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))

    recs = ctx.writer.records
    trunk = [r for r in recs if not r.root_forced]
    branches = [r for r in recs if r.root_forced]
    assert len(trunk) == 1 and trunk[0].label_from_dp == 0
    # 2 roots x 2 actions (1 own = open drawer 1, 1 tail = look)
    assert len(branches) == 4
    assert {b.action_source for b in branches} == {"own", "tail"}
    for b in branches:
        assert b.label_from_dp == b.root_dp == b.root_turn
        assert b.cont_len == 5 - b.root_turn - 1
    s = ctx.stats
    assert s.games_done == 1 and s.trunks == 1 and s.roots == 2 and s.branches_run == 4
    assert s.roots_with_contrast == 2  # open drawer 1 returned 1.0 vs trunk 0.0
    assert s.accept_by_block_source == {"sampled": 2, "elicited": 2}
    assert s.root_labels_by_verb_class == {"open": 2, "look": 2}
    # node JSON + done marker
    nodes = sorted((tmp_path / "nodes").rglob("t*.json"))
    assert len(nodes) == 2
    node = json.loads(nodes[0].read_text())
    assert node["schema"] == 1 and len(node["chosen"]) == 2 and node["chosen"][0]["g"] in (0.0, 1.0)
    assert (tmp_path / "done" / "pps-x__trial_1.json").exists()
    assert (tmp_path / "collect_stats.json").exists()


def test_skip_and_drop_keep_game(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"open drawer 1": 1.0}, drop_branch={"look"})
    cfg = _cfg(tmp_path, roots_per_trunk=1)
    node_env = _FakeNodeEnv(own=["go to desk 1", "open drawer 1"], fail_actions={"open drawer 1"})
    ctx = _make_ctx(tmp_path, cfg, node_env)
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_2/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    s = ctx.stats
    assert s.branches_skipped == {"no_congruent_block": 1}
    assert s.reject_reasons == {"argmax": 1, "tau": 1, "leak": 1}
    assert s.branches_dropped == 1 and s.branches_run == 0
    assert s.games_done == 1
    node = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())
    kinds = {c["action"]: c["block_source"] for c in node["chosen"]}
    assert kinds["open drawer 1"] == "none"
    assert node["replay_ok"] is False


def test_resume_skips_done_game(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    cfg = _cfg(tmp_path)
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "pps-x__trial_1.json").write_text("{}")
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1"]))
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    assert ctx.stats.games_skipped_resume == 1 and ctx.writer.records == []


def test_two_stage_allocation(tmp_path, monkeypatch):
    # own[1] = open drawer 1 is the probe action; it flips the return -> pivotal
    _install_fakes(monkeypatch, {"open drawer 1": 1.0, "look": 0.0})
    cfg = _cfg(tmp_path, allocation="two_stage", stage1_roots=2, stage2_roots=1, branches_per_root=2, n_uniform=1)
    node_env = _FakeNodeEnv(own=["go to desk 1", "open drawer 1"])
    ctx = _make_ctx(tmp_path, cfg, node_env)
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_3/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    branches = [r for r in ctx.writer.records if r.root_forced]
    stages = sorted(b.allocation_stage for b in branches)
    # stage 1: one probe branch at each of 2 candidates; stage 2: K-1 = 1 more at 1 pivotal root
    assert stages == [1, 1, 2]
    s = ctx.stats
    assert s.stage1_roots == 2 and s.stage2_roots == 1
    # the stage-2 branch is the tail action, since the probe action is excluded
    stage2 = [b for b in branches if b.allocation_stage == 2][0]
    assert stage2.action_source == "tail"
    # node scored once per root (cache hit at stage 2)
    assert len(node_env.scored) == 2


def test_holdout_tagging(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    cfg = _cfg(tmp_path, holdout_games=1, roots_per_trunk=1)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]), n_games=2)
    ex = lambda i: {"prompt": [], "info": {"game_file": f"/d/train/pps-x/trial_{i}/game.tw-pddl"}, "example_id": i, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, ex(0)))
    asyncio.run(bc.collect_game(ctx, 1, ex(1)))
    by_game = {}
    for r in ctx.writer.records:
        by_game.setdefault(r.game_id, set()).add(r.holdout)
    assert by_game["pps-x__trial_0"] == {False}
    assert by_game["pps-x__trial_1"] == {True}


def _gold_example(i=4):
    return {"prompt": [], "info": {"game_file": f"/d/train/pps-x/trial_{i}/game.tw-pddl"}, "example_id": 0, "task": "t"}


def test_all_actions_with_replays(tmp_path, monkeypatch):
    """Gold-set mode: every admissible action at a root, the played one included,
    each replayed; the node row carries the replays and their mean."""
    _install_fakes(monkeypatch, {"open drawer 1": 1.0, "look": 0.0, "go to desk 1": 0.0})

    def _no_pick(*a, **k):
        raise AssertionError("the pick rule must not be consulted when all_actions is on")

    monkeypatch.setattr(bc, "_pick_actions", _no_pick)
    cfg = _cfg(tmp_path, roots_per_trunk=1, all_actions=True, replays=3, max_branches_per_game=100)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example()))

    branches = [r for r in ctx.writer.records if r.root_forced]
    # 1 root x 3 admissible actions x 3 replays
    assert len(branches) == 9
    assert {b.action_source for b in branches} == {"taken", "own", "tail"}
    ids = [b.branch_id for b in branches]
    assert len(set(ids)) == 9 and all(i.rsplit("-", 1)[1] in ("r0", "r1", "r2") for i in ids)
    node = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())
    assert [c["action"] for c in node["chosen"]] == ["go to desk 1", "open drawer 1", "look"]
    by = {c["action"]: c for c in node["chosen"]}
    assert by["go to desk 1"]["source"] == "taken"
    assert by["open drawer 1"]["g"] == 1.0 and by["open drawer 1"]["g_replays"] == [1.0, 1.0, 1.0]
    assert len(by["look"]["branch_ids"]) == 3 and by["look"]["branch_id"] == by["look"]["branch_ids"][0]
    s = ctx.stats
    assert s.branches_run == 9 and s.roots_with_contrast == 1
    assert s.accept_by_action_source == {"taken": 1, "own": 1, "tail": 1}
    assert json.loads((tmp_path / "done" / "pps-x__trial_4.json").read_text())["branches"] == 9


def test_replay_mean_skips_dropped_replays(tmp_path, monkeypatch):
    """One replay wins, one loses, one is dropped: the row's g is the mean of the two that ran."""
    _install_fakes(monkeypatch, {})

    async def fake_run_rollout(env, client, example, model_name, sampling_args, max_retries):
        seed = (example.get("info") or {}).get("branch_seed")
        if seed is None:
            return _trunk_out(T=3, reward=0.0)
        rep = seed["branch_id"].rsplit("-", 1)[1]
        if seed["forced_action"] == "open drawer 1" and rep == "r2":
            raise RuntimeError("wedged")
        win = seed["forced_action"] == "open drawer 1" and rep == "r0"
        return {"reward": 1.0 if win else 0.0, "is_truncated": False, "error": None, "trajectory": [],
                "completion": [], "example_id": 0, "sampling_args": {"temperature": 0.8}}

    monkeypatch.setattr(bc, "run_rollout", fake_run_rollout)
    cfg = _cfg(tmp_path, roots_per_trunk=1, branches_per_root=1, n_uniform=0, replays=3)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(5)))
    node = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())
    (row,) = node["chosen"]
    assert row["action"] == "open drawer 1"
    assert row["g_replays"] == [1.0, 0.0, None] and row["g"] == pytest.approx(0.5)
    assert node["replay_ok"] is False
    assert ctx.stats.branches_run == 2 and ctx.stats.branches_dropped == 1
    assert ctx.stats.roots_with_contrast == 1  # 0.5 differs from the trunk's 0.0


def test_replays_count_against_the_game_budget(tmp_path, monkeypatch):
    """The per-game cap counts episodes: with 3 replays and a cap of 4 only one action fits."""
    _install_fakes(monkeypatch, {"open drawer 1": 1.0})
    cfg = _cfg(tmp_path, roots_per_trunk=1, all_actions=True, replays=3, max_branches_per_game=4)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(6)))
    assert ctx.stats.branches_run == 3
    assert ctx.stats.branches_skipped == {"budget": 2}


def test_second_pass_replays_the_played_action_and_what_beat_it(tmp_path, monkeypatch):
    """First pass: 'open drawer 1' wins, the played action and 'look' lose. Only the
    played action and the action that beat it get the extra episodes."""
    _install_fakes(monkeypatch, {"open drawer 1": 1.0, "look": 0.0, "go to desk 1": 0.0})
    cfg = _cfg(tmp_path, roots_per_trunk=1, all_actions=True, replays=2, replays_extra=3, max_branches_per_game=100)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(9)))
    node = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())
    by = {c["action"]: c for c in node["chosen"]}
    assert len(by["go to desk 1"]["g_replays"]) == 5 and by["go to desk 1"]["extra_replays"] == 3
    assert len(by["open drawer 1"]["g_replays"]) == 5 and by["open drawer 1"]["g"] == 1.0
    assert len(by["look"]["g_replays"]) == 2 and "extra_replays" not in by["look"]
    ids = [b.branch_id for b in ctx.writer.records if b.root_forced]
    assert len(ids) == len(set(ids)) == 12  # 3 actions x 2, then 2 actions x 3
    assert by["open drawer 1"]["branch_ids"][-1].endswith("-r4")
    assert ctx.stats.branches_run == 12
    assert json.loads((tmp_path / "done" / "pps-x__trial_9.json").read_text())["branches"] == 12


def test_no_second_pass_when_nothing_beat_the_played_action(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"go to desk 1": 1.0, "open drawer 1": 1.0, "look": 0.0})
    cfg = _cfg(tmp_path, roots_per_trunk=1, all_actions=True, replays=2, replays_extra=3, max_branches_per_game=100)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(10)))
    assert ctx.stats.branches_run == 6  # a tie with the played action is not a lead
    node = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())
    assert all("extra_replays" not in c for c in node["chosen"])


def test_second_pass_needs_the_lead_and_respects_the_budget(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})

    async def fake_run_rollout(env, client, example, model_name, sampling_args, max_retries):
        seed = (example.get("info") or {}).get("branch_seed")
        if seed is None:
            return _trunk_out(T=3, reward=0.0)
        # 'open drawer 1' wins one of its two first replays: a lead of 0.5 over the played action
        win = seed["forced_action"] == "open drawer 1" and seed["branch_id"].endswith("-r0")
        return {"reward": 1.0 if win else 0.0, "is_truncated": False, "error": None, "trajectory": [],
                "completion": [], "example_id": 0, "sampling_args": {"temperature": 0.8}}

    monkeypatch.setattr(bc, "run_rollout", fake_run_rollout)
    common = dict(roots_per_trunk=1, all_actions=True, replays=2, replays_extra=2)

    strict = _make_ctx(tmp_path / "a", _cfg(tmp_path / "a", extra_lead=0.75, max_branches_per_game=100, **common),
                       _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(strict, 0, _gold_example(11)))
    assert strict.stats.branches_run == 6  # the lead of 0.5 is below 0.75

    loose = _make_ctx(tmp_path / "b", _cfg(tmp_path / "b", extra_lead=0.5, max_branches_per_game=100, **common),
                      _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(loose, 0, _gold_example(12)))
    assert loose.stats.branches_run == 10  # the played action and the contender, two more each
    node = json.loads(next((tmp_path / "b" / "nodes").rglob("t*.json")).read_text())
    by = {c["action"]: c for c in node["chosen"]}
    assert by["open drawer 1"]["g_replays"] == [1.0, 0.0, 0.0, 0.0] and by["open drawer 1"]["g"] == 0.25

    tight = _make_ctx(tmp_path / "c", _cfg(tmp_path / "c", extra_lead=0.5, max_branches_per_game=8, **common),
                      _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(tight, 0, _gold_example(13)))
    # 6 first-pass episodes, room for one more pair only: the played action gets it
    assert tight.stats.branches_run == 8 and tight.stats.branches_skipped == {"budget": 1}


def test_single_replay_keeps_the_old_ids_and_rows(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"open drawer 1": 1.0})
    cfg = _cfg(tmp_path, roots_per_trunk=1, branches_per_root=1, n_uniform=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(7)))
    (branch,) = [r for r in ctx.writer.records if r.root_forced]
    assert branch.branch_id.endswith("-b0")
    (row,) = json.loads(next((tmp_path / "nodes").rglob("t*.json")).read_text())["chosen"]
    assert "g_replays" not in row and "branch_ids" not in row and row["g"] == 1.0


def test_game_start_skips_earlier_games(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    cfg = _cfg(tmp_path, roots_per_trunk=1, game_start=1, holdout_games=1)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]), n_games=2)
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(0)))
    assert ctx.writer.records == [] and ctx.stats.trunks == 0 and not (tmp_path / "done").exists()
    asyncio.run(bc.collect_game(ctx, 1, _gold_example(1)))
    assert {r.game_id for r in ctx.writer.records} == {"pps-x__trial_1"}
    assert all(r.holdout for r in ctx.writer.records)


def test_game_end_stops_the_slice(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    cfg = _cfg(tmp_path, roots_per_trunk=1, game_start=1, game_end=2, holdout_games=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]), n_games=3)
    for i in range(3):
        asyncio.run(bc.collect_game(ctx, i, _gold_example(i)))
    assert {r.game_id for r in ctx.writer.records} == {"pps-x__trial_1"}
    assert sorted(p.stem for p in (tmp_path / "done").glob("*.json")) == ["pps-x__trial_1"]
    inputs = [_gold_example(i) for i in range(4)]
    assert bc.slice_game_ids(inputs, 1, 3) == ["pps-x__trial_1", "pps-x__trial_2"]
    assert bc.slice_game_ids(inputs, 1, 99) == bc.slice_game_ids(inputs, 1)
    with pytest.raises(ValueError, match="game_end"):
        _cfg(tmp_path, game_start=2, game_end=2)


def test_denied_games_guard(tmp_path):
    inputs = [_gold_example(i) for i in range(4)]
    bc.check_denied_games(inputs, 0, None)  # no list: nothing to check
    held = tmp_path / "holdout.json"
    held.write_text(json.dumps({"game_ids": ["pps-x__trial_3"]}))
    with pytest.raises(ValueError, match=r"1 of the 4 game\(s\).*pps-x__trial_3"):
        bc.check_denied_games(inputs, 0, held)
    # the slice decides: ending before the listed game is fine
    bc.check_denied_games(inputs, 0, held, 3)
    # and the allowed list reads the same slice
    with pytest.raises(ValueError, match=r"3 of the 3 game\(s\)"):
        bc.check_allowed_games(inputs, 0, held, 3)
    bc.check_allowed_games(inputs, 3, held, 4)


def test_allowed_games_guard(tmp_path):
    inputs = [_gold_example(i) for i in range(4)]
    assert bc.slice_game_ids(inputs, 2) == ["pps-x__trial_2", "pps-x__trial_3"]
    bc.check_allowed_games(inputs, 2, None)  # no list: nothing to check
    as_list = tmp_path / "ids.json"
    as_list.write_text(json.dumps(["pps-x__trial_2", "pps-x__trial_3"]))
    bc.check_allowed_games(inputs, 2, as_list)
    as_holdout = tmp_path / "holdout.json"  # the trainer's holdout.json shape
    as_holdout.write_text(json.dumps({"game_ids": ["pps-x__trial_3"]}))
    with pytest.raises(ValueError, match=r"1 of the 2 game\(s\).*pps-x__trial_2"):
        bc.check_allowed_games(inputs, 2, as_holdout)
    # the slice decides: the same list is fine once the collect starts at the listed game
    bc.check_allowed_games(inputs, 3, as_holdout)


def test_even_allocation_drives_the_roots(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    cfg = _cfg(tmp_path, allocation="even", roots_per_trunk=2, branches_per_root=1, n_uniform=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    asyncio.run(bc.collect_game(ctx, 0, _gold_example(8)))
    # the fake trunk has 5 turns: candidates 0..3, two slices -> their centres 1 and 3
    assert sorted(r.root_turn for r in ctx.writer.records if r.root_forced) == [1, 3]
    node = json.loads(next((tmp_path / "nodes").rglob("t1.json")).read_text())
    assert node["allocation"] == "even"


def test_gold_options_validation(tmp_path):
    with pytest.raises(ValueError, match="all_actions"):
        _cfg(tmp_path, all_actions=True, allocation="two_stage", stage1_roots=2, stage2_roots=1)
    assert _cfg(tmp_path, all_actions=True, allocation="even").allocation == "even"
    with pytest.raises(ValueError, match="replays_extra"):
        _cfg(tmp_path, replays_extra=2)
    with pytest.raises(ValueError, match="replays"):
        _cfg(tmp_path, replays=5, max_branches_per_game=4)


def test_branch_env_args_injects_branch_mode(tmp_path):
    cfg = _cfg(tmp_path)
    args = bc._branch_env_args(cfg)
    assert args["branch_mode"] is True and args["num_reasoning_blocks"] == 1
    assert args["tau"] == 0.5 and "{action}" in args["elicit_prompt"]


def test_records_roundtrip_through_writer(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"open drawer 1": 1.0})
    cfg = _cfg(tmp_path, roots_per_trunk=1, branches_per_root=1, n_uniform=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    path = tmp_path / "records.msgpack"
    with RecordWriter(path) as w:
        for r in ctx.writer.records:
            w.write(r)
    back = list(read_branch_records(path))
    assert [r.root_forced for r in back] == [False, True]
    assert back[1].block_source == "sampled" and back[1].action_source == "own"


def test_terminal_forced_action_keeps_branch(tmp_path, monkeypatch):
    """A forced action that ends the game returns an empty continuation (no
    completion); the branch is still recorded with cont_len 0."""
    _install_fakes(monkeypatch, {})

    async def fake_run_rollout(env, client, example, model_name, sampling_args, max_retries):
        seed = (example.get("info") or {}).get("branch_seed")
        if seed is None:
            return _trunk_out(T=3, reward=0.0)
        return {"reward": 1.0, "is_truncated": False, "error": None, "trajectory": [],
                "completion": [], "example_id": 0, "sampling_args": {"temperature": 0.8}}

    monkeypatch.setattr(bc, "run_rollout", fake_run_rollout)
    cfg = _cfg(tmp_path, roots_per_trunk=1, branches_per_root=1, n_uniform=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    branches = [r for r in ctx.writer.records if r.root_forced]
    assert len(branches) == 1
    assert branches[0].cont_len == 0 and branches[0].reward == 1.0
    assert ctx.stats.roots_with_contrast == 1


def test_replay_failure_is_a_skip(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})

    async def fake_run_rollout(env, client, example, model_name, sampling_args, max_retries):
        seed = (example.get("info") or {}).get("branch_seed")
        if seed is None:
            return _trunk_out(T=3, reward=0.0)
        return {"reward": 0.0, "is_truncated": False, "error": "replay invariant failed: obs mismatch",
                "trajectory": [], "completion": [], "example_id": 0, "sampling_args": {"temperature": 0.8}}

    monkeypatch.setattr(bc, "run_rollout", fake_run_rollout)
    cfg = _cfg(tmp_path, roots_per_trunk=1, branches_per_root=1, n_uniform=0)
    ctx = _make_ctx(tmp_path, cfg, _FakeNodeEnv(own=["go to desk 1", "open drawer 1"]))
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))
    assert ctx.stats.replay_failures == 1
    assert ctx.stats.branches_skipped == {"replay": 1}
    assert ctx.stats.branches_dropped == 0


def test_pick_actions_resolves_real_module():
    """The driver's lazy import reaches the verifiers env package's pick_actions.
    On a dev box without the alfworld env package installed, the in-repo module
    is put on sys.path; skipped when neither exists."""
    import sys
    try:
        bc._pick_actions(["b"], ["a", "b"], "a", 1, 0, random.Random(0))
    except ImportError:
        dev = Path("/home/ubuntu/verifiers/environments/alfworld")
        if not (dev / "branching.py").exists():
            pytest.skip("alfworld env package not importable here")
        sys.path.insert(0, str(dev))
    picks = bc._pick_actions(["b", "a"], ["a", "b", "c", "d"], "a", 3, 1, random.Random(0))
    assert len(picks) == 3
    assert picks[0] == ("b", "own")
    assert {src for _, src in picks} == {"own", "tail"}
    assert all(a != "a" for a, _ in picks)


# ---------------------------------------------------------------------------
# Weak acceptance (D-17): theta reaches the env, weak blocks are recorded
# ---------------------------------------------------------------------------


class _WeakNodeEnv(_FakeNodeEnv):
    """Own actions are strong (the base stub); tail actions come back weakly
    accepted with an attempt log, as synthesize_block does under theta < tau."""

    async def synthesize_block(self, client, model, prompt, admissible, action, node, block_args):
        if action in node.own:
            return await super().synthesize_block(client, model, prompt, admissible, action, node, block_args)
        syn = _Syn("why", "elicited", 0.05, 3, ["argmax", "argmax", "argmax"], None,
                   f"<think>why</think><action>{action}</action>")
        syn.attempt_log = [
            {"stage": "sampled", "block_idx": 0, "p": 0.001, "rank": 3, "argmax": node.own[0]},
            {"stage": "elicited", "reasoning": "why", "p": 0.05, "rank": 2, "argmax": node.own[0],
             "reason": "argmax", "accepted": "weak"},
        ]
        return syn


def test_weak_acceptance_is_recorded(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {"open drawer 1": 1.0, "look": 0.0})
    cfg = _cfg(tmp_path, theta=0.01)
    node_env = _WeakNodeEnv(own=["go to desk 1", "open drawer 1"])
    ctx = _make_ctx(tmp_path, cfg, node_env)
    example = {"prompt": [], "info": {"game_file": "/d/train/pps-x/trial_1/game.tw-pddl"}, "example_id": 0, "task": "t"}
    asyncio.run(bc.collect_game(ctx, 0, example))

    s = ctx.stats
    assert s.accept_by_action_source == {"own": 2, "tail": 2}
    assert s.accept_weak_by_action_source == {"tail": 2}
    tail = [r for r in ctx.writer.records if r.root_forced and r.action_source == "tail"]
    assert len(tail) == 2 and all(r.p_accept == 0.05 for r in tail)
    node = json.loads(sorted((tmp_path / "nodes").rglob("t*.json"))[0].read_text())
    by_src = {c["source"]: c for c in node["chosen"]}
    assert by_src["tail"]["strong"] is False and by_src["own"]["strong"] is True
    assert [e["stage"] for e in by_src["tail"]["attempt_log"]] == ["sampled", "elicited"]
    assert by_src["tail"]["attempt_log"][1]["accepted"] == "weak"
    assert by_src["own"]["attempt_log"] == []


def test_branch_env_args_passes_theta(tmp_path):
    assert bc._branch_env_args(_cfg(tmp_path))["theta"] is None
    assert bc._branch_env_args(_cfg(tmp_path, theta=0.02))["theta"] == 0.02


def test_branch_env_args_passes_elicit_variants(tmp_path):
    args = bc._branch_env_args(_cfg(tmp_path))
    assert args["elicit_variants"] == [] and args["elicit_exhaustive"] is False
    args = bc._branch_env_args(_cfg(tmp_path, elicit_variants=["ALT {action}"], elicit_exhaustive=True, n_elicit=4))
    assert args["elicit_variants"] == ["ALT {action}"] and args["elicit_exhaustive"] is True
    assert args["n_elicit"] == 4


# ---------------------------------------------------------------------------
# Resume hygiene: unfinished games are dropped before appending
# ---------------------------------------------------------------------------


def _rec(game, branch=None):
    from prime_rl.value_warmstart.dataset_io import BranchRecord

    return BranchRecord(samples=[], reward=0.0, is_truncated=False, game_id=game, trunk_id=f"{game}-trunk",
                        branch_id=branch, root_forced=branch is not None, root_dp=0 if branch else None,
                        root_turn=0 if branch else None)


def _layout(tmp_path, records, done, nodes):
    from prime_rl.value_warmstart.dataset_io import RecordWriter

    with RecordWriter(tmp_path / "records.msgpack") as w:
        for r in records:
            w.write(r)
    (tmp_path / "done").mkdir()
    for g in done:
        (tmp_path / "done" / f"{g}.json").write_text("{}")
    for g in nodes:
        d = tmp_path / "nodes" / g / f"{g}-trunk"
        d.mkdir(parents=True)
        (d / "t0.json").write_text("{}")


def test_drop_unfinished_games_removes_partial_output(tmp_path):
    from prime_rl.value_warmstart.dataset_io import read_branch_records

    recs = [_rec("g1"), _rec("g1", "b0"), _rec("g2"), _rec("g2", "b0"), _rec("g1", "b1"), _rec("g3")]
    _layout(tmp_path, recs, done=["g1"], nodes=["g1", "g2", "g3"])
    # a writer killed mid-frame leaves a truncated tail
    with open(tmp_path / "records.msgpack", "ab") as f:
        f.write(b"\x40\x00\x00\x00partial")

    out = bc.drop_unfinished_games(tmp_path)
    assert (out["kept"], out["dropped"], out["truncated"]) == (3, 3, True)
    assert out["dropped_games"] == ["g2", "g3"] and out["node_dirs_removed"] == 2
    kept = list(read_branch_records(tmp_path / "records.msgpack"))
    assert [(r.game_id, r.branch_id) for r in kept] == [("g1", None), ("g1", "b0"), ("g1", "b1")]
    assert sorted(p.name for p in (tmp_path / "nodes").iterdir()) == ["g1"]
    assert not list(tmp_path.glob("*.compact"))


def test_drop_unfinished_games_is_a_noop_when_clean(tmp_path):
    recs = [_rec("g1"), _rec("g1", "b0"), _rec("g2")]
    _layout(tmp_path, recs, done=["g1", "g2"], nodes=["g1", "g2"])
    before = (tmp_path / "records.msgpack").read_bytes()
    out = bc.drop_unfinished_games(tmp_path)
    assert (out["kept"], out["dropped"], out["truncated"], out["node_dirs_removed"]) == (3, 0, False, 0)
    assert (tmp_path / "records.msgpack").read_bytes() == before
    assert not list(tmp_path.glob("*.compact"))


def test_removing_a_done_marker_discards_that_game(tmp_path):
    from prime_rl.value_warmstart.dataset_io import read_branch_records

    recs = [_rec("g1"), _rec("g2"), _rec("g2", "b0")]
    _layout(tmp_path, recs, done=["g1", "g2"], nodes=["g1", "g2"])
    (tmp_path / "done" / "g2.json").unlink()
    out = bc.drop_unfinished_games(tmp_path)
    assert out["dropped"] == 2 and out["dropped_games"] == ["g2"]
    assert [r.game_id for r in read_branch_records(tmp_path / "records.msgpack")] == ["g1"]
