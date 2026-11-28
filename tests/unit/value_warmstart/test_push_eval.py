"""Bulletproof warm start, Round 0: the push-direction score on a gold test set
(gold_states / truth_of / rule_advantages / score_states / summarize), with stub
value networks and stub policy probabilities."""

import json

import pytest

from prime_rl.advantage_server.action_advantage import cleared_regrets, difference_advantage
from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.dataset_io import BranchRecord
from prime_rl.value_warmstart.push_eval import (
    GoldState,
    gold_states,
    label_noise,
    node_pi_hat_fn,
    pi_hat_from_node,
    rule_advantages,
    score_states,
    summarize,
    truth_of,
)

ADM = ["go to desk 1", "open drawer 1", "go to dresser 1", "take pen 1"]


def _sample(actions, prompt=(1, 2)):
    dps = [
        DecisionPoint(
            response_start=2 * k, response_end=2 * k + 2, admissible_actions=list(adm), executed_action_idx=ei
        )
        for k, (adm, ei) in enumerate(actions)
    ]
    clen = 2 * len(actions)
    return TrainingSample(
        prompt_ids=list(prompt),
        prompt_mask=[False] * len(prompt),
        completion_ids=list(range(10, 10 + clen)),
        completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen,
        completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


def _trunk(game="g1", reward=0.0, holdout=True):
    # the policy plays "go to desk 1" (idx 0) at turns 0 and 1
    return BranchRecord(
        samples=[_sample([(ADM, 0), (ADM, 0), (ADM, 1)])],
        reward=reward,
        is_truncated=False,
        game_id=game,
        trunk_id=f"{game}-t0",
        holdout=holdout,
    )


def _branch(action_idx, reward, bid, game="g1", t=1, src="tail", holdout=True):
    return BranchRecord(
        samples=[_sample([(ADM, 0), (ADM, action_idx), (ADM, 2)])],
        reward=reward,
        is_truncated=False,
        label_from_dp=t,
        root_dp=t,
        root_forced=True,
        game_id=game,
        trunk_id=f"{game}-t0",
        branch_id=bid,
        root_turn=t,
        action_source=src,
        block_source="elicited",
        p_accept=0.7,
        holdout=holdout,
    )


def _replays(action_idx, rewards, game="g1", t=1):
    return [_branch(action_idx, r, f"{game}-t{t}-b{action_idx}-r{j}", game=game, t=t) for j, r in enumerate(rewards)]


# ---------------------------------------------------------------------------
# records -> states
# ---------------------------------------------------------------------------


def test_gold_states_groups_replays_by_root_and_action():
    recs = [_trunk(reward=0.0)] + _replays(0, [0, 0, 1]) + _replays(1, [1, 1, 1]) + _replays(2, [0, 0, 0])
    (st,) = gold_states(recs)
    assert (st.game_id, st.trunk_id, st.turn) == ("g1", "g1-t0", 1)
    assert st.a_taken == "go to desk 1" and st.admissible == ADM
    # the root context: prompt + completion up to the root response
    assert st.o_ids == [1, 2, 10, 11]
    # the trunk's own outcome is one more replay of the played action
    assert st.returns["go to desk 1"] == [0.0, 0.0, 1.0, 0.0]
    assert st.returns["open drawer 1"] == [1.0, 1.0, 1.0]
    assert "take pen 1" not in st.returns
    (st2,) = gold_states(recs, include_trunk=False)
    assert st2.returns["go to desk 1"] == [0.0, 0.0, 1.0]


def test_gold_states_needs_the_trunk_and_honours_holdout():
    assert gold_states(_replays(1, [1, 1])) == []
    recs = [_trunk(holdout=False)] + [_branch(1, 1.0, "b", holdout=False)]
    assert gold_states(recs, holdout_only=True) == []
    assert len(gold_states(recs)) == 1


# ---------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------


def _state(returns, a_taken="go to desk 1", game="g1", turn=1):
    return GoldState(
        game_id=game,
        trunk_id=f"{game}-t0",
        turn=turn,
        o_ids=[1],
        admissible=list(ADM),
        a_taken=a_taken,
        returns=returns,
    )


def test_truth_kinds():
    all_lost = {a: [0.0, 0.0, 0.0, 0.0] for a in ADM}
    beaten = truth_of(_state({**all_lost, "open drawer 1": [1.0, 1.0, 0.0, 1.0]}))
    assert beaten.kind == "beaten" and beaten.better == ["open drawer 1"] and beaten.rate_taken == 0.0
    fine = truth_of(_state({**all_lost, "go to desk 1": [1.0, 1.0, 1.0, 1.0]}))
    assert fine.kind == "fine" and fine.better == [] and fine.coverage == 1.0
    # a lead of 0.25: neither clearly better nor within the fine gap -> unclear
    unclear = truth_of(_state({**all_lost, "open drawer 1": [1.0, 0.0, 0.0, 0.0]}))
    assert unclear.kind == "unclear"
    # ...and fine once the fine gap allows a quarter
    assert truth_of(_state({**all_lost, "open drawer 1": [1.0, 0.0, 0.0, 0.0]}), fine_gap=0.25).kind == "fine"


def test_truth_needs_replays_of_the_played_action_and_coverage():
    assert truth_of(_state({"open drawer 1": [1.0, 1.0]})).kind == "unclear"
    assert truth_of(_state({"go to desk 1": [0.0], "open drawer 1": [1.0, 1.0]})).rate_taken is None
    # two of four actions labelled: nothing beats the played one, but coverage is 0.5
    partial = _state({"go to desk 1": [1.0, 1.0], "open drawer 1": [0.0, 0.0]})
    assert truth_of(partial).kind == "unclear"
    assert truth_of(partial, min_coverage=0.5).kind == "fine"
    # a clearly better action is enough whatever the coverage
    assert truth_of(_state({"go to desk 1": [0.0, 0.0], "open drawer 1": [1.0, 1.0]})).kind == "beaten"


def test_label_noise_counts_mixed_actions_and_flips():
    st = _state({"go to desk 1": [0.0, 0.0, 0.0, 0.0], "open drawer 1": [1.0, 0.0, 1.0, 0.0], "take pen 1": [1.0]})
    noise = label_noise([st])
    assert noise["actions_with_replays"] == 2 and noise["mixed_share"] == 0.5
    # desk: 0 of 6 pairs differ; drawer: 4 of 6
    assert noise["flip_rate"] == pytest.approx(4 / 12)
    assert noise["wins_of_replays"] == {"0/4": 1, "2/4": 1}


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def test_rule_advantages_matches_the_server_functions():
    q, v, pi = [0.9, 0.2, 0.45, 0.1], 0.3, [0.02, 0.9, 0.03, 0.05]
    adv = rule_advantages(q, v, pi, margin=0.10, anchor=0.3)
    r = cleared_regrets(q, v, 0.10)
    assert adv == [difference_advantage(r[i], sum(r), pi[i], 0.3) for i in range(4)]
    assert adv[0] > 0.05 and adv[1] < -0.05
    # no cleared regret -> no push on anything
    assert rule_advantages([0.3, 0.35, 0.2, 0.1], 0.3, pi, margin=0.10, anchor=0.3) == [0.0] * 4


def test_pi_hat_from_node_uses_sampled_blocks_and_mass():
    node = {
        "admissible": ADM,
        "blocks": [
            {"row": [1.0, 0.0, 0.0, 0.0], "mass": 1.0},  # block 0: the greedy one, not a sample
            {"row": [0.5, 0.5, 0.0, 0.0], "mass": 0.8},
            {"row": [0.0, 1.0, 0.0, 0.0], "mass": 1.0},
        ],
    }
    pi = pi_hat_from_node(node, ADM)
    assert pi == pytest.approx([0.2, 0.7, 0.0, 0.0])
    # a different action order is matched by text
    assert pi_hat_from_node(node, list(reversed(ADM))) == pytest.approx([0.0, 0.0, 0.7, 0.2])
    assert pi_hat_from_node(node, ADM + ["look"]) is None
    assert pi_hat_from_node({"admissible": ADM, "blocks": []}, ADM) is None


def test_node_pi_hat_fn_reads_the_collect_layout(tmp_path):
    st = _state({})
    d = tmp_path / st.game_id / st.trunk_id
    d.mkdir(parents=True)
    (d / "t1.json").write_text(json.dumps({"admissible": ADM, "blocks": [{"row": [0.25] * 4, "mass": 1.0}]}))
    assert node_pi_hat_fn(tmp_path)(st) == pytest.approx([0.25] * 4)
    assert node_pi_hat_fn(tmp_path / "missing")(st) is None


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

PEAKED = [0.94, 0.02, 0.02, 0.02]  # the policy is sure of "go to desk 1"


def _values(q):
    return lambda o_ids, admissible: (0.3, [q[a] for a in admissible])


def test_score_states_verdicts():
    lost = [0.0, 0.0, 0.0, 0.0]
    beaten = _state({a: lost for a in ADM} | {"open drawer 1": [1.0, 1.0, 1.0, 1.0]}, game="g1")
    fine = _state({a: lost for a in ADM} | {"go to desk 1": [1.0, 1.0, 1.0, 1.0]}, game="g2")
    states = [beaten, fine]

    # a good network: only "open drawer 1" clears the margin, and only where it is better
    def good(o_ids, admissible):
        return 0.3, [0.9 if (a == "open drawer 1" and o_ids == [1]) else 0.3 for a in admissible]

    beaten.o_ids, fine.o_ids = [1], [2]
    rows = score_states(states, good, lambda st: PEAKED)
    by = {r["game_id"]: r for r in rows}
    assert by["g1"]["kind"] == "beaten" and by["g1"]["catch"] is True and by["g1"]["lift"] is True
    assert by["g1"]["lifted"] == "open drawer 1" and by["g1"]["harm"] is None
    assert by["g2"]["kind"] == "fine" and by["g2"]["harm"] is False and by["g2"]["catch"] is None
    assert by["g2"]["adv"] == [0.0] * 4 and by["g2"]["lifted"] is None

    # a network with a common offset: every action looks regretted everywhere
    rows = score_states(states, _values({a: 0.6 for a in ADM}), lambda st: PEAKED)
    by = {r["game_id"]: r for r in rows}
    assert by["g2"]["harm"] is True  # pushed off a fine action
    assert by["g1"]["catch"] is True  # ...and off the beaten one, for the wrong reason
    # every other action is lifted by the same amount: the offset carries no information
    adv = by["g1"]["adv"]
    assert adv[1] == pytest.approx(adv[2]) == pytest.approx(adv[3]) and adv[1] > 0.05

    # a network that lifts the wrong action
    rows = score_states(
        states,
        _values({"go to desk 1": 0.3, "open drawer 1": 0.3, "go to dresser 1": 0.9, "take pen 1": 0.3}),
        lambda st: PEAKED,
    )
    by = {r["game_id"]: r for r in rows}
    assert by["g1"]["lifted"] == "go to dresser 1" and by["g1"]["lift"] is False and by["g1"]["lift_weak"] is False


def test_score_states_without_policy_probabilities():
    st = _state({a: [0.0, 0.0] for a in ADM})
    (row,) = score_states([st], _values({a: 0.3 for a in ADM}), lambda st: None)
    assert row["scored"] is False and row["kind"] == "fine"


def test_summarize_rates_intervals_and_verdict():
    rows = []
    for g in range(30):
        rows.append(
            {
                "game_id": f"f{g}",
                "kind": "fine",
                "scored": True,
                "harm": False,
                "catch": None,
                "lift": None,
                "lift_weak": None,
            }
        )
        rows.append(
            {
                "game_id": f"b{g}",
                "kind": "beaten",
                "scored": True,
                "harm": None,
                "catch": True,
                "lift": True,
                "lift_weak": True,
            }
        )
    good = summarize(rows, bootstrap=200)
    assert good["harm"]["rate"] == 0.0 and good["catch"]["rate"] == 1.0 and good["lift"]["rate"] == 1.0
    assert good["harm"]["n"] == 30 and good["kinds"] == {"fine": 30, "beaten": 30, "unclear": 0}
    assert good["passed"] is True and good["helpful_pushes"] == 30 and good["harmful_pushes"] == 0
    assert good["helpful_to_harmful"] is None

    for r in rows[:20:2]:
        r["harm"] = True  # 10 of 30 fine states pushed down
    bad = summarize(rows, bootstrap=200)
    assert bad["harm"]["rate"] == pytest.approx(1 / 3) and bad["harm"]["verdict"] == "FAIL"
    assert bad["passed"] is False and bad["helpful_to_harmful"] == pytest.approx(3.0)
    lo, hi = bad["harm"]["interval95"]
    assert 0.1 < lo < 1 / 3 < hi < 0.6


def test_summarize_point_estimate_on_the_line_needs_the_interval():
    # catch is 0.9 on 10 states: above the 0.8 line, but the interval is not
    rows = [
        {
            "game_id": f"b{g}",
            "kind": "beaten",
            "scored": True,
            "harm": None,
            "catch": g != 0,
            "lift": True,
            "lift_weak": True,
        }
        for g in range(10)
    ]
    s = summarize(rows, bootstrap=400)
    assert s["catch"]["rate"] == 0.9 and s["catch"]["verdict"] == "FAIL"
    assert s["harm"]["verdict"] == "NOT AVAILABLE" and s["passed"] is False
    assert summarize(rows, bootstrap=0)["catch"]["verdict"] == "NO INTERVAL"


def test_push_eval_truth_only_reads_a_records_file(tmp_path):
    """The pilot's read-out: no networks, a records file in, kinds and label noise out."""
    from prime_rl.value_warmstart.dataset_io import RecordWriter
    from prime_rl.value_warmstart.push_eval import PushEvalConfig, push_eval

    recs = [_trunk(reward=0.0)]
    for idx, rewards in ((0, [0, 0, 0, 0]), (1, [1, 1, 1, 1]), (2, [0, 1, 0, 0]), (3, [0, 0, 0, 0])):
        recs += _replays(idx, rewards)
    path = tmp_path / "records.msgpack"
    with RecordWriter(path) as w:
        for r in recs:
            w.write(r)
    out = tmp_path / "gate" / "truth.json"
    payload = push_eval(PushEvalConfig(dataset=path, truth_only=True, out=out))
    assert json.loads(out.read_text()) == payload
    assert payload["truth"]["states"] == 1 and payload["truth"]["kinds"] == {"fine": 0, "beaten": 1, "unclear": 0}
    assert payload["truth"]["actions_per_state"] == 4 and payload["truth"]["labelled_per_state"] == 4
    noise = payload["label_noise"]
    # four actions with replays (the played one has five: four forced and the trunk); one is mixed
    assert noise["actions_with_replays"] == 4 and noise["mixed_share"] == 0.25
    assert "summary" not in payload and "states" not in payload
    with pytest.raises(ValueError, match="value_state"):
        push_eval(PushEvalConfig(dataset=path, out=out))


def test_fine_states_split_into_live_and_dead():
    from prime_rl.value_warmstart.push_eval import fine_kind

    def state(returns, turn=1):
        return GoldState("g", "g-t0", turn, [1], list(ADM), ADM[0], returns)

    dead = truth_of(state({a: [0.0, 0.0] for a in ADM}))
    live = truth_of(state({ADM[0]: [1.0, 1.0], ADM[1]: [1.0, 0.0], ADM[2]: [0.0, 0.0], ADM[3]: [0.0, 0.0]}))
    beaten = truth_of(state({ADM[0]: [0.0, 0.0], ADM[1]: [1.0, 1.0], ADM[2]: [0.0, 0.0], ADM[3]: [0.0, 0.0]}))
    assert (dead.kind, fine_kind(dead)) == ("fine", "dead")
    assert (live.kind, fine_kind(live)) == ("fine", "live")
    assert (beaten.kind, fine_kind(beaten)) == ("beaten", None)


def test_breakdowns_split_harm_and_read_by_turn():
    from prime_rl.value_warmstart.push_eval import breakdowns

    def row(turn, kind, fk=None, **verdicts):
        base = {"harm": None, "catch": None, "lift": None}
        return {"game_id": "g", "turn": turn, "kind": kind, "fine_kind": fk, "scored": True, **base, **verdicts}

    rows = [
        row(0, "fine", "live", harm=True),
        row(0, "fine", "live", harm=False),
        row(2, "beaten", catch=True, lift=False),
        row(7, "fine", "dead", harm=False),
        row(7, "unclear"),
        {"game_id": "g", "turn": 7, "kind": "fine", "fine_kind": "dead", "scored": False},
    ]
    b = breakdowns(rows)
    assert b["harm_live"] == {"rate": 0.5, "n": 2} and b["harm_dead"] == {"rate": 0.0, "n": 1}
    assert list(b["by_turn"]) == ["0", "2", "7"]
    assert b["by_turn"]["2"]["catch"] == {"rate": 1.0, "n": 1} and b["by_turn"]["2"]["lift"]["rate"] == 0.0
    t7 = b["by_turn"]["7"]
    assert t7["states"] == 3 and t7["dead"] == 2 and t7["kinds"]["unclear"] == 1 and t7["harm"]["n"] == 1


def test_push_eval_reads_several_gold_folders(tmp_path):
    """Two collect folders of one gold set, read without a merged copy: every
    state carries its folder's name, and a game in two folders is refused."""
    from prime_rl.value_warmstart.dataset_io import RecordWriter
    from prime_rl.value_warmstart.push_eval import PushEvalConfig, load_gold, push_eval, rows_by_source

    def folder(name, game):
        d = tmp_path / name
        d.mkdir()
        recs = [_trunk(game=game, reward=0.0)]
        for idx, rewards in ((0, [0, 0]), (1, [1, 1]), (2, [0, 0]), (3, [0, 0])):
            recs += [_branch(idx, float(x), f"{game}-b{idx}-r{j}", game=game) for j, x in enumerate(rewards)]
        with RecordWriter(d / "records.msgpack") as w:
            for r in recs:
                w.write(r)
        return d

    a, b = folder("heldout", "g1"), folder("valid-unseen", "g2")
    cfg = PushEvalConfig(gold_dirs=[a, b], truth_only=True, out=tmp_path / "truth.json")
    states, nodes = load_gold(cfg)
    assert [(s.game_id, s.source) for s in states] == [("g1", "heldout"), ("g2", "valid-unseen")]
    assert nodes == {"heldout": a / "nodes", "valid-unseen": b / "nodes"}
    payload = push_eval(cfg)
    assert payload["truth"]["states"] == 2 and payload["truth"]["kinds"]["beaten"] == 2
    assert payload["dataset"] == [str(a), str(b)]
    rows = [{"game_id": "g1", "source": "heldout"}, {"game_id": "g2", "source": "valid-unseen"}]
    assert set(rows_by_source(rows, {})) == {"heldout", "valid-unseen"}

    twice = folder("again", "g1")
    with pytest.raises(ValueError, match="g1 is in both 'heldout' and 'again'"):
        load_gold(PushEvalConfig(gold_dirs=[a, twice], truth_only=True, out=tmp_path / "x.json"))
    with pytest.raises(ValueError, match="exactly one"):
        PushEvalConfig(truth_only=True, out=tmp_path / "x.json")
    with pytest.raises(ValueError, match="exactly one"):
        PushEvalConfig(dataset=a / "records.msgpack", gold_dirs=[b], truth_only=True, out=tmp_path / "x.json")


def test_node_probabilities_per_source(tmp_path):
    node = {"admissible": list(ADM), "blocks": [{"row": [1, 0, 0, 0]}, {"row": [0.5, 0.5, 0, 0], "mass": 1.0}]}
    p = tmp_path / "b" / "nodes" / "g" / "g-t0"
    p.mkdir(parents=True)
    (p / "t1.json").write_text(json.dumps(node))
    fn = node_pi_hat_fn({"b": tmp_path / "b" / "nodes"})
    st = GoldState("g", "g-t0", 1, [1], list(ADM), ADM[0], {}, "b")
    assert fn(st) == [0.5, 0.5, 0.0, 0.0]
    st.source = "elsewhere"
    assert fn(st) is None


# ---------------------------------------------------------------------------
# the task statement in the context
# ---------------------------------------------------------------------------


def _trunk_two_samples(game="g1"):
    # the window drops its oldest message after turn 1: turns 0-1 are in the
    # first sample, turns 2-3 in a second one with its own (shorter) prompt
    return BranchRecord(
        samples=[_sample([(ADM, 0), (ADM, 0)]), _sample([(ADM, 0), (ADM, 1)], prompt=(7, 8, 9))],
        reward=0.0,
        is_truncated=False,
        game_id=game,
        trunk_id=f"{game}-t0",
        holdout=True,
    )


def _late_replays(action_idx, rewards, game="g1"):
    # forced at turn 2, the first turn after the drop
    return [
        BranchRecord(
            samples=[_sample([(ADM, 0), (ADM, 0), (ADM, action_idx)])],
            reward=float(r),
            is_truncated=False,
            label_from_dp=2,
            root_dp=2,
            root_forced=True,
            game_id=game,
            trunk_id=f"{game}-t0",
            branch_id=f"{game}-t2-b{action_idx}-r{j}",
            root_turn=2,
            action_source="tail",
            holdout=True,
        )
        for j, r in enumerate(rewards)
    ]


def _two_sided_records(game="g1"):
    recs = [_trunk_two_samples(game)]
    for idx, rewards in ((0, [0, 0]), (1, [1, 1]), (2, [0, 0]), (3, [0, 0])):
        recs += _replays(idx, rewards, game=game, t=1)
        recs += _late_replays(idx, [0, 0], game=game)
    return recs


def test_gold_states_count_the_window_drops_before_a_turn():
    early, late = gold_states(_two_sided_records())
    assert (early.turn, early.window_drops) == (1, 0)
    assert (late.turn, late.window_drops) == (2, 1)
    # the later state's context is the second sample's
    assert early.o_ids == [1, 2, 10, 11] and late.o_ids == [7, 8, 9]
    rows = score_states([early, late], _values({a: 0.3 for a in ADM}), lambda st: PEAKED)
    assert [r["task_in_context"] for r in rows] == [True, False]
    assert [r["window_drops"] for r in rows] == [0, 1]
    assert [(r["kind"], r["fine_kind"]) for r in rows] == [("beaten", None), ("fine", "dead")]


def test_the_line_counts_where_the_task_is_in_the_context():
    from prime_rl.value_warmstart.push_eval import line_verdict, rows_by_task

    def row(g, in_ctx, **verdicts):
        base = {"harm": None, "catch": None, "lift": None, "lift_weak": None}
        kind = "fine" if "harm" in verdicts else "beaten"
        return {"game_id": g, "kind": kind, "scored": True, "task_in_context": in_ctx, **base, **verdicts}

    rows = []
    for g in range(40):
        # task in the context: never harms, always catches and lifts
        rows.append(row(f"a{g}", True, harm=False))
        rows.append(row(f"a{g}", True, catch=True, lift=True))
        # task gone: quiet in the fine states, blind in the beaten ones
        rows.append(row(f"a{g}", False, harm=False))
        rows.append(row(f"a{g}", False, catch=False, lift=False))
    split = rows_by_task(rows)
    assert len(split["in_context"]) == 80 and len(split["gone"]) == 80
    inc, gone = summarize(split["in_context"], seed=1), summarize(split["gone"], seed=1)
    verdict = line_verdict(inc, gone)
    assert verdict["basis"] == "task_in_context" and verdict["passed"] is True
    assert verdict["catch"]["rate"] == 1.0 and verdict["lift"]["verdict"] == "PASS"
    assert verdict["quiet"]["rate"] == 0.0 and verdict["quiet_passed"] is True
    # all states together would fail: the blind half drags catch and lift down
    assert summarize(rows, seed=1)["passed"] is False
    # a pushy network after the drop fails the quiet check and nothing else
    loud = [dict(r, harm=True) if (not r["task_in_context"] and r["kind"] == "fine") else r for r in rows]
    s2 = rows_by_task(loud)
    v2 = line_verdict(summarize(s2["in_context"], seed=1), summarize(s2["gone"], seed=1))
    assert v2["passed"] is True and v2["quiet_passed"] is False
    # a row written before the split existed counts as in the context
    old = {k: x for k, x in rows[3].items() if k != "task_in_context"}
    assert rows_by_task([old]) == {"in_context": [old], "gone": []}
    # no state after a drop: the quiet check does not apply
    none = line_verdict(inc, summarize([], seed=1))
    assert none["quiet_passed"] is None and none["passed"] is True


def test_push_eval_truth_only_counts_states_on_both_sides(tmp_path):
    from prime_rl.value_warmstart.dataset_io import RecordWriter
    from prime_rl.value_warmstart.push_eval import PushEvalConfig, push_eval

    path = tmp_path / "records.msgpack"
    with RecordWriter(path) as w:
        for r in _two_sided_records():
            w.write(r)
    payload = push_eval(PushEvalConfig(dataset=path, truth_only=True, out=tmp_path / "truth.json"))
    by = payload["truth"]["by_task"]
    assert by["in_context"] == {
        "states": 1,
        "kinds": {"fine": 0, "beaten": 1, "unclear": 0},
        "fine_live": 0,
        "fine_dead": 0,
    }
    assert by["gone"] == {"states": 1, "kinds": {"fine": 1, "beaten": 0, "unclear": 0}, "fine_live": 0, "fine_dead": 1}


def test_push_eval_scores_both_sides_and_judges_the_line_in_context(tmp_path, monkeypatch):
    """The whole scoring path with a stub network: the payload carries the split,
    the verdict reads the in-context side, and the after-drop side is the quiet check."""
    import transformers

    from prime_rl.value_warmstart import push_eval as pe
    from prime_rl.value_warmstart.dataset_io import RecordWriter

    gold = tmp_path / "heldout"
    gold.mkdir()
    with RecordWriter(gold / "records.msgpack") as w:
        for g in ("g1", "g2", "g3"):
            for r in _two_sided_records(g):
                w.write(r)
    node = {"admissible": list(ADM), "blocks": [{"row": [0.94, 0.02, 0.02, 0.02]}]}
    for g in ("g1", "g2", "g3"):
        d = gold / "nodes" / g / f"{g}-t0"
        d.mkdir(parents=True)
        for t in (1, 2):
            (d / f"t{t}.json").write_text(json.dumps(node))

    class Backbone:
        def __init__(self, base, lora_config=None, polyak_tau=0.0):
            pass

        def to(self, **kw):
            return self

    def value_fn(o_ids, admissible):
        # "open drawer 1" looks good everywhere: right before the drop, a wrong push after it
        return 0.3, [0.9 if a == "open drawer 1" else 0.3 for a in admissible]

    monkeypatch.setattr(pe, "ValueNetworkBackbone", Backbone)
    monkeypatch.setattr(pe, "load_warm_start", lambda backbone, path: None)
    monkeypatch.setattr(pe, "build_value_fn", lambda backbone, tokenizer, device: value_fn)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object()))

    out = tmp_path / "gate" / "push.json"
    payload = pe.push_eval(
        pe.PushEvalConfig(
            gold_dirs=[gold],
            value_state=tmp_path / "value_state.pt",
            model={"base_model_name": "stub", "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.0}},
            bootstrap=200,
            out=out,
        )
    )
    assert json.loads(out.read_text()) == payload and payload["schema"] == 2
    assert [r["task_in_context"] for r in payload["states"]] == [True, False] * 3
    inc, gone = payload["by_task"]["in_context"], payload["by_task"]["gone"]
    assert (inc["states"], gone["states"]) == (3, 3)
    assert inc["catch"]["rate"] == 1.0 and inc["lift"]["rate"] == 1.0 and inc["harm"]["rate"] is None
    assert gone["harm"]["rate"] == 1.0 and gone["harm"]["n"] == 3 and gone["catch"]["rate"] is None
    verdict = payload["verdict"]
    assert verdict["basis"] == "task_in_context"
    assert verdict["catch"] == inc["catch"] and verdict["quiet"] == gone["harm"]
    assert verdict["passed"] is inc["passed"] and verdict["quiet_passed"] is False
    # the all-states summary is still there, and the per-folder split has an in-context twin
    assert payload["summary"]["harm"]["n"] == 3 and payload["summary"]["catch"]["n"] == 3
    assert payload["by_source"]["heldout"]["states"] == 6
    assert payload["by_source_in_context"]["heldout"]["states"] == 3
