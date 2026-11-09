"""Branched dataset Phase 3: contrast_pairs / score_pairs (gate G-A) with a stub Q+."""

from prime_rl.transport.types import DecisionPoint, TrainingSample
from prime_rl.value_warmstart.contrast_eval import contrast_pairs, score_pairs
from prime_rl.value_warmstart.dataset_io import BranchRecord


def _sample(actions, prompt=(1, 2)):
    """One decision point per (admissible, executed) pair, 2 completion tokens each."""
    dps = [
        DecisionPoint(response_start=2 * k, response_end=2 * k + 2, admissible_actions=list(adm), executed_action_idx=ei)
        for k, (adm, ei) in enumerate(actions)
    ]
    clen = 2 * len(actions)
    return TrainingSample(
        prompt_ids=list(prompt), prompt_mask=[False] * len(prompt),
        completion_ids=list(range(10, 10 + clen)), completion_mask=[True] * clen,
        completion_logprobs=[0.0] * clen, completion_temperatures=[1.0] * clen,
        decision_points=dps,
    )


ADM = ["go to desk 1", "open drawer 1", "go to dresser 1", "take pen 1"]


def _trunk(game="g1", reward=0.0, holdout=True):
    # turn 0 executes "go to desk 1", turn 1 executes "go to desk 1" (idx 0)
    return BranchRecord(
        samples=[_sample([(ADM, 0), (ADM, 0), (ADM, 1)])], reward=reward, is_truncated=False,
        game_id=game, trunk_id=f"{game}-t0", holdout=holdout,
    )


def _branch(action_idx, reward, bid, game="g1", t=1, block="sampled", src="own", holdout=True, truncated=False):
    return BranchRecord(
        samples=[_sample([(ADM, 0), (ADM, action_idx), (ADM, 2)])], reward=reward, is_truncated=truncated,
        label_from_dp=t, root_dp=t, root_forced=True, game_id=game, trunk_id=f"{game}-t0",
        branch_id=bid, root_turn=t, action_source=src, block_source=block, p_accept=0.7, holdout=holdout,
    )


def test_enumerates_pairs_with_differing_mean_returns():
    recs = [
        _trunk(reward=0.0),
        _branch(1, 1.0, "b1"),                 # open drawer -> 1
        _branch(2, 0.0, "b2", block="elicited", src="tail"),  # go to dresser -> 0
        _branch(3, 1.0, "b3", src="tail"),     # take pen -> 1 (reward 1 but...)
        _branch(3, 1.0, "b4", src="tail", truncated=True),  # ...truncated -> 0; mean 0.5
    ]
    pairs = contrast_pairs(recs)
    keys = {(p.a_i, p.a_j) for p in pairs}
    # desk (0.0, trunk) vs drawer (1.0), desk vs pen (0.5), drawer vs dresser (0.0),
    # drawer vs pen, dresser vs pen; dresser vs desk both 0.0 -> excluded
    assert ("go to desk 1", "open drawer 1") in keys
    assert ("go to desk 1", "take pen 1") in keys
    assert ("go to dresser 1", "open drawer 1") in keys
    assert ("open drawer 1", "take pen 1") in keys
    assert ("go to dresser 1", "take pen 1") in keys
    assert ("go to desk 1", "go to dresser 1") not in keys
    assert len(pairs) == 5
    p = next(p for p in pairs if (p.a_i, p.a_j) == ("go to desk 1", "open drawer 1"))
    assert p.g_i == 0.0 and p.g_j == 1.0
    assert p.meta_i["provenance"] == ["trunk"]
    assert p.meta_j["block_source"] == ["sampled"] and p.meta_j["action_source"] == ["own"]
    # o is the root context: prompt + completion up to the root response
    assert p.o_ids == [1, 2, 10, 11]
    pen = next(p for p in pairs if p.a_j == "take pen 1" and p.a_i == "go to desk 1")
    assert pen.g_j == 0.5 and pen.meta_j["n_episodes"] == 2


def test_holdout_only_filters_and_requires_trunk():
    recs = [_trunk(holdout=False), _branch(1, 1.0, "b1", holdout=False)]
    assert contrast_pairs(recs) == []
    assert len(contrast_pairs(recs, holdout_only=False)) == 1
    assert len(contrast_pairs(recs, holdout_ids={"g1"})) == 1
    # a branch whose trunk record is missing still pairs with other branches only
    recs2 = [_branch(1, 1.0, "b1"), _branch(2, 0.0, "b2")]
    pairs = contrast_pairs(recs2)
    assert len(pairs) == 1 and {pairs[0].a_i, pairs[0].a_j} == {"open drawer 1", "go to dresser 1"}


def test_score_pairs_accuracy_and_splits():
    recs = [_trunk(reward=0.0), _branch(1, 1.0, "b1"), _branch(2, 0.0, "b2", block="elicited", src="tail")]
    pairs = contrast_pairs(recs)
    assert len(pairs) == 2  # desk vs drawer, dresser vs drawer
    # a perfect Q+: drawer high, others low
    q_good = {"open drawer 1": 0.9, "go to desk 1": 0.1, "go to dresser 1": 0.2}
    res = score_pairs(pairs, lambda o, a: q_good[a])
    assert res["n_pairs"] == 2 and res["accuracy"] == 1.0 and res["n_roots"] == 1
    assert res["by_pair_kind"]["trunk-branch"]["n"] == 1
    assert res["by_pair_kind"]["branch-branch"]["n"] == 1
    assert res["by_block_source"]["sampled"]["n"] == 2
    assert res["by_block_source"]["elicited"]["n"] == 1
    assert res["by_action_source"]["tail"]["accuracy"] == 1.0
    # a flat Q+ orders nothing correctly (ties count as wrong)
    res_flat = score_pairs(pairs, lambda o, a: 0.0)
    assert res_flat["accuracy"] == 0.0
    # max_pairs truncates deterministically
    res_one = score_pairs(pairs, lambda o, a: q_good[a], max_pairs=1)
    assert res_one["n_pairs"] == 1
    row = res["pairs"][0]
    assert set(row) >= {"game_id", "trunk_id", "root_turn", "a_i", "a_j", "g_i", "g_j", "q_i", "q_j", "correct"}


def test_q_fn_called_once_per_root_action():
    recs = [_trunk(reward=0.0), _branch(1, 1.0, "b1"), _branch(2, 0.0, "b2")]
    pairs = contrast_pairs(recs)
    calls = []

    def q(o, a):
        calls.append(a)
        return {"open drawer 1": 1.0}.get(a, 0.0)

    score_pairs(pairs, q)
    assert sorted(calls) == ["go to desk 1", "go to dresser 1", "open drawer 1"]


def test_min_p_accept_filter_and_p_band_split():
    weak = _branch(2, 0.0, "b2", block="elicited", src="tail")  # go to dresser -> 0
    weak.p_accept = 0.03
    recs = [_trunk(reward=0.0), _branch(1, 1.0, "b1"), weak]  # trunk desk 0, drawer 1 (p 0.7)
    pairs = contrast_pairs(recs)
    # (desk, drawer) and (dresser, drawer); desk vs dresser tie on return
    assert len(pairs) == 2
    out = score_pairs(pairs, lambda o_ids, a: 1.0 if a == "open drawer 1" else 0.0)
    assert out["by_p_band"]["p>=0.5"] == {"n": 2, "accuracy": 1.0}
    assert out["by_p_band"]["0.01<=p<0.1"] == {"n": 1, "accuracy": 1.0}
    kept = contrast_pairs(recs, min_p_accept=0.1)
    assert len(kept) == 1
    assert {kept[0].a_i, kept[0].a_j} == {"go to desk 1", "open drawer 1"}
