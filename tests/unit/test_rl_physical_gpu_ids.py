"""PRIME_RL_PHYSICAL_GPU_IDS: launcher GPU colocation override (2026-11-07) and
MIG-UUID device tokens (2026-11-08: SLURM exports MIG UUIDs in CUDA_VISIBLE_DEVICES)."""

from __future__ import annotations

import pytest

from prime_rl.entrypoints.rl import _parse_device_tokens, get_physical_gpu_ids

MIG0 = "MIG-dddb4a3d-bff4-569c-a98f-871be13705c1"
MIG1 = "MIG-1edb0d93-aad9-5ffa-bcd5-3ce6fd5fc5ef"


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


def test_mig_uuids_pass_through_in_override(monkeypatch):
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", f"{MIG0},{MIG1},{MIG0},{MIG1}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG0},{MIG1}")
    assert get_physical_gpu_ids() == [MIG0, MIG1, MIG0, MIG1]


def test_mig_uuids_in_cuda_visible_devices_without_override(monkeypatch):
    monkeypatch.delenv("PRIME_RL_PHYSICAL_GPU_IDS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG0},{MIG1}")
    assert get_physical_gpu_ids() == [MIG0, MIG1]


def test_parse_mixed_and_gpu_uuid():
    assert _parse_device_tokens("0,GPU-3538a582-d700-7296-8770-c64be76db22b, 2") == [
        0, "GPU-3538a582-d700-7296-8770-c64be76db22b", 2,
    ]


def test_launcher_mapping_with_override_is_four_local_to_two_physical(monkeypatch):
    """The mapping rl_local builds from the override: local 0..3 -> [A, B, A, B], and the
    per-subprocess CUDA_VISIBLE_DEVICES string each receives."""
    monkeypatch.setenv("PRIME_RL_PHYSICAL_GPU_IDS", f"{MIG0},{MIG1},{MIG0},{MIG1}")
    physical = get_physical_gpu_ids()
    total_requested = 4  # infer 1 + trainer 1 + adv server 1 + adv trainer 1
    assert total_requested <= len(physical)
    mapping = {local: physical[local] for local in range(total_requested)}
    assert mapping == {0: MIG0, 1: MIG1, 2: MIG0, 3: MIG1}
    assert ",".join(map(str, [mapping[0]])) == MIG0  # what the inference subprocess gets
    assert ",".join(map(str, [mapping[3]])) == MIG1  # what the Advantage Trainer gets
