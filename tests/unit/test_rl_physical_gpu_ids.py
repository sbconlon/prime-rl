"""PRIME_RL_PHYSICAL_GPU_IDS: launcher GPU colocation override (2026-11-07)."""

from __future__ import annotations

import pytest

from prime_rl.entrypoints.rl import get_physical_gpu_ids


def test_override_allows_duplicates_and_wins_over_cuda_visible(monkeypatch):
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", "0,1,0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert get_physical_gpu_ids() == [0, 1, 0, 1]


def test_cuda_visible_devices_without_override(monkeypatch):
    monkeypatch.delenv("PRIME_RL_PHYSICAL_GPU_IDS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert get_physical_gpu_ids() == [2, 3]


def test_empty_override_falls_through(monkeypatch):
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", "  ")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert get_physical_gpu_ids() == [1]


def test_override_tolerates_spaces_and_trailing_comma(monkeypatch):
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", " 0, 1 ,0,1, ")
    assert get_physical_gpu_ids() == [0, 1, 0, 1]


def test_bad_token_raises(monkeypatch):
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", "0,x")
    with pytest.raises(ValueError):
        get_physical_gpu_ids()


def test_launcher_mapping_with_override_is_four_local_to_two_physical(monkeypatch):
    """The mapping rl_local builds from the override: local 0..3 -> [0,1,0,1]."""
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", "0,1,0,1")
    physical = get_physical_gpu_ids()
    total_requested = 4  # infer 1 + trainer 1 + adv server 1 + adv trainer 1
    assert total_requested <= len(physical)
    mapping = {local: physical[local] for local in range(total_requested)}
    assert mapping == {0: 0, 1: 1, 2: 0, 3: 1}
    assert sorted(set(mapping.values())) == [0, 1]
