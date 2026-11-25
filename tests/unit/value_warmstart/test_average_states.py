"""Bulletproof warm start, Round 0: averaging value_state.pt files of one run."""

import pytest
import torch

from prime_rl.value_warmstart.average_states import AverageStatesConfig, average_states, average_trainable


def _payload(step, a, b, dtype=torch.float32):
    return {
        "step": step,
        "trainable_state_dict": {
            "lora_A": torch.full((2, 3), a, dtype=dtype),
            "head.weight": torch.full((1, 4), b, dtype=dtype),
        },
        "optimizer_state_dict": {"state": {}},
    }


def test_average_is_the_mean_per_tensor():
    out = average_trainable([_payload(10, 1.0, 0.0), _payload(20, 3.0, 6.0), _payload(30, 5.0, 0.0)])
    assert out["step"] == 30 and out["averaged_from_steps"] == [10, 20, 30]
    assert torch.equal(out["trainable_state_dict"]["lora_A"], torch.full((2, 3), 3.0))
    assert torch.equal(out["trainable_state_dict"]["head.weight"], torch.full((1, 4), 2.0))
    # the optimizer state is not carried: an average has none
    assert "optimizer_state_dict" not in out


def test_average_of_one_is_that_one_and_inputs_are_untouched():
    p = _payload(7, 2.0, -1.0)
    before = {k: v.clone() for k, v in p["trainable_state_dict"].items()}
    out = average_trainable([p])
    for k, v in before.items():
        assert torch.equal(out["trainable_state_dict"][k], v) and torch.equal(p["trainable_state_dict"][k], v)
        assert out["trainable_state_dict"][k] is not p["trainable_state_dict"][k]


def test_average_keeps_the_dtype_and_sums_in_float32():
    a, b = _payload(1, 1.0, 0.0, torch.bfloat16), _payload(2, 2.0, 0.0, torch.bfloat16)
    out = average_trainable([a, b])
    t = out["trainable_state_dict"]["lora_A"]
    assert t.dtype == torch.bfloat16 and torch.equal(t, torch.full((2, 3), 1.5, dtype=torch.bfloat16))


def test_average_refuses_different_networks():
    other = _payload(1, 1.0, 0.0)
    other["trainable_state_dict"]["extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="same parameters"):
        average_trainable([_payload(1, 1.0, 0.0), other])
    reshaped = _payload(1, 1.0, 0.0)
    reshaped["trainable_state_dict"]["lora_A"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        average_trainable([_payload(1, 1.0, 0.0), reshaped])
    with pytest.raises(ValueError, match="nothing"):
        average_trainable([])


def test_average_states_writes_a_loadable_warm_start(tmp_path):
    paths = []
    for i, a in enumerate((0.0, 2.0)):
        p = tmp_path / f"m{i}" / "value_state.pt"
        p.parent.mkdir()
        torch.save(_payload(100 * (i + 1), a, 1.0), p)
        paths.append(p)
    out_path = tmp_path / "avg" / "trainer" / "value_state.pt"
    average_states(AverageStatesConfig(inputs=paths, out=out_path))
    back = torch.load(out_path, map_location="cpu", weights_only=False)
    assert back["step"] == 200 and back["averaged_from"] == [str(p) for p in paths]
    assert torch.equal(back["trainable_state_dict"]["lora_A"], torch.full((2, 3), 1.0))
    # the shape load_warm_start reads: a dict with "trainable_state_dict"
    assert set(back["trainable_state_dict"]) == {"lora_A", "head.weight"}
