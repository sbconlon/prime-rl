"""Unit tests for eval scheduling logic - specifically the range check
that detects when ckpt_step jumps over eval interval boundaries."""

from prime_rl.orchestrator.eval_utils import compute_eval_ckpt_step


def test_exact_hit():
    result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=24, last_eval_step=0, interval=25)
    assert result == 25


def test_jump_over_interval():
    result = compute_eval_ckpt_step(ckpt_step=26, prev_ckpt_step=24, last_eval_step=0, interval=25)
    assert result == 25


def test_no_interval_crossed():
    result = compute_eval_ckpt_step(ckpt_step=23, prev_ckpt_step=22, last_eval_step=0, interval=25)
    assert result is None


def test_base_model_eval_at_step_0():
    result = compute_eval_ckpt_step(
        ckpt_step=0, prev_ckpt_step=-1, last_eval_step=-1, interval=25, eval_base_model=True
    )
    assert result == 0


def test_base_model_eval_disabled():
    result = compute_eval_ckpt_step(
        ckpt_step=0, prev_ckpt_step=-1, last_eval_step=-1, interval=25, eval_base_model=False
    )
    assert result is None


def test_no_double_eval():
    result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=24, last_eval_step=25, interval=25)
    assert result is None


def test_no_change_in_ckpt_step():
    result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=25, last_eval_step=0, interval=25)
    assert result is None


def test_multiple_intervals_crossed():
    result = compute_eval_ckpt_step(ckpt_step=76, prev_ckpt_step=24, last_eval_step=0, interval=25)
    assert result == 75


def test_second_interval():
    result = compute_eval_ckpt_step(ckpt_step=50, prev_ckpt_step=49, last_eval_step=25, interval=25)
    assert result == 50


def test_jump_across_second_interval():
    result = compute_eval_ckpt_step(ckpt_step=51, prev_ckpt_step=48, last_eval_step=25, interval=25)
    assert result == 50


def test_production_scenario_step25_skipped():
    """Reproduces the bug from run c14miuyha2yhxkw1z3eqgyub."""
    result = compute_eval_ckpt_step(ckpt_step=26, prev_ckpt_step=24, last_eval_step=0, interval=25)
    assert result == 25


def test_production_scenario_step50_exact():
    result = compute_eval_ckpt_step(ckpt_step=50, prev_ckpt_step=49, last_eval_step=26, interval=25)
    assert result == 50


def test_simulate_full_run():
    ckpt_steps = [0, 0, 3, 5, 10, 15, 20, 24, 26, 30, 35, 40, 48, 51, 60, 70, 74, 76]
    interval = 25
    last_eval_step = -1
    prev_ckpt_step = -1
    eval_triggered_at = []

    for ckpt_step in ckpt_steps:
        result = compute_eval_ckpt_step(ckpt_step, prev_ckpt_step, last_eval_step, interval)
        if result is not None:
            eval_triggered_at.append(result)
            last_eval_step = ckpt_step
        prev_ckpt_step = ckpt_step

    assert eval_triggered_at == [0, 25, 50, 75]


# ---------------------------------------------------------------------------
# training-probe (Phase 3): the probe block reuses compute_eval_ckpt_step with its
# own bookkeeping (last_probe_step, probe_base_model, skip_probe_on_resume).
# ---------------------------------------------------------------------------


def _simulate_probe(ckpt_steps, *, interval, probe_base_model=True, resume_skip_at=None):
    last_probe_step = -1
    prev_ckpt_step = -1
    if resume_skip_at is not None:
        # orchestrator resume branch with skip_probe_on_resume=True and no eval configured:
        # prev_ckpt_step one behind (eval's else branch), last_probe_step pinned at the resume step
        prev_ckpt_step = resume_skip_at - 1
        last_probe_step = resume_skip_at
    fired = []
    for ckpt_step in ckpt_steps:
        result = compute_eval_ckpt_step(
            ckpt_step=ckpt_step,
            prev_ckpt_step=prev_ckpt_step,
            last_eval_step=last_probe_step,
            interval=interval,
            eval_base_model=probe_base_model,
        )
        if result is not None:
            last_probe_step = ckpt_step
            fired.append((ckpt_step, result))
        prev_ckpt_step = ckpt_step
    return fired


def test_probe_gating_mirrors_eval():
    # base model at 0, then every 10, jumping over a boundary (18 -> 23 probes "20" at 23),
    # never twice for one ckpt_step
    steps = [0, 0, 3, 7, 10, 10, 14, 18, 23, 30, 31]
    assert _simulate_probe(steps, interval=10) == [(0, 0), (10, 10), (23, 20), (30, 30)]
    # probe_base_model=False drops the ckpt-0 round only
    assert _simulate_probe(steps, interval=10, probe_base_model=False) == [(10, 10), (23, 20), (30, 30)]
    # resume at 20 with skip_probe_on_resume: the redundant round at 20 is skipped, 30 still fires
    assert _simulate_probe([20, 20, 25, 30], interval=10, resume_skip_at=20) == [(30, 30)]
    # resume at 20 without skip: prev one behind lets the resumed step probe once
    assert _simulate_probe([20, 20, 25, 30], interval=10)[:1] == [(20, 20)]
