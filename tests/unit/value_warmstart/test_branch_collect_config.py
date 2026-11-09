"""Branched dataset Phase 0: BranchCollectConfig defaults and validation."""

from pathlib import Path

import pytest

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.configs.value_warmstart import (
    ELICIT_DEFAULT,
    LEAK_DEFAULT,
    BranchCollectConfig,
)


def _cfg(**overrides):
    base = dict(
        model_name="sft",
        env=EnvConfig(id="alfworld"),
        client=ClientConfig(),
        output_path=Path("unused.msgpack"),
    )
    base.update(overrides)
    return BranchCollectConfig(**base)


def test_defaults_match_plan():
    c = _cfg()
    assert (c.roots_per_trunk, c.branches_per_root, c.n_uniform, c.node_blocks) == (6, 4, 2, 8)
    assert c.allocation == "stratified"
    assert c.early_tilt == 0.5
    assert (c.stage1_roots, c.stage2_roots) == (10, 4)
    assert c.tau == 0.5 and c.n_elicit == 3
    assert "{action}" in ELICIT_DEFAULT and c.elicit_prompt == ELICIT_DEFAULT
    assert c.leak_patterns == LEAK_DEFAULT
    assert c.holdout_games == 50 and c.resume is True


def test_n_uniform_bounded_by_k():
    with pytest.raises(ValueError):
        _cfg(branches_per_root=2, n_uniform=3)


def test_stage2_bounded_by_stage1():
    with pytest.raises(ValueError):
        _cfg(stage1_roots=3, stage2_roots=4)


def test_node_blocks_at_least_two():
    with pytest.raises(ValueError):
        _cfg(node_blocks=1)


def test_tau_open_interval():
    with pytest.raises(ValueError):
        _cfg(tau=1.0)
