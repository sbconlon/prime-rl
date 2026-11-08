"""training-probe Phase 0 -- additive diagnostics fields on the AdvServer wire types.

The request gains a trailing `return_diagnostics: bool = False`; the response
gains trailing `diagnostics` / `weight_step` defaulting to None. msgspec does
NOT trim trailing defaults from array_like structs (omit_defaults applies to the
dict encoding), so the wire arrays grow by one / two trailing elements. What
must hold: old-shape bytes decode into the new structs with the defaults, and
the new encoding is exactly the old array plus the trailing defaults -- no
field is reordered or renamed. Client and server ship from the same checkout,
so no mixed-version pairing occurs in practice.
"""

from typing import Literal

import msgspec

from prime_rl.orchestrator.advantage_server_client import (
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
    DecisionPointDiagnostics,
    PairedSample,
)
from prime_rl.transport.types import AdvantageTrainingSample, TrainingSample


# Pre-probe shapes, reproduced locally so the test does not depend on git history.
class _LegacyRequest(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    samples: list[TrainingSample]
    episodic_reward: float
    is_terminal: bool
    algorithm: Literal["ppo", "arm"]


class _LegacyResponse(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    paired_samples: list[PairedSample]


def _sample(n: int = 3) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=list(range(10, 10 + n)),
        completion_mask=[True] * n,
        completion_logprobs=[-0.1] * n,
        completion_temperatures=[1.0] * n,
    )


def _paired(n: int = 3) -> PairedSample:
    s = _sample(n)
    llm = msgspec.structs.replace(s, advantages=[0.5] * n)
    adv = AdvantageTrainingSample(
        prompt_ids=list(s.prompt_ids),
        prompt_mask=list(s.prompt_mask),
        completion_ids=list(s.completion_ids),
        completion_mask=list(s.completion_mask),
    )
    return PairedSample(llm_sample=llm, advantage_sample=adv)


def _as_array(obj) -> list:
    """Decode a struct's msgpack bytes as a plain list (positional view)."""
    return msgspec.msgpack.decode(msgspec.msgpack.encode(obj))


def test_request_flag_defaults_false_and_is_legacy_array_plus_flag():
    new = ComputeAdvantagesRequest(samples=[_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm")
    legacy = _LegacyRequest(samples=[_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm")
    assert new.return_diagnostics is False
    # the training path (flag off) puts the old array on the wire plus one
    # trailing False: positions of every existing field are unchanged
    assert _as_array(new) == _as_array(legacy) + [False]


def test_request_default_decodes_from_old_shape():
    legacy = _LegacyRequest(samples=[_sample()], episodic_reward=0.0, is_terminal=False, algorithm="ppo")
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(legacy), type=ComputeAdvantagesRequest)
    assert decoded.return_diagnostics is False
    assert decoded.algorithm == "ppo"
    assert decoded.samples[0].completion_ids == [10, 11, 12]


def test_legacy_structs_decode_new_bytes():
    """msgspec ignores trailing array elements beyond a struct's fields, so an
    old server would still decode a new client's request (flag off or on) and an
    old client a new server's response. Pins the forward direction too."""
    req = ComputeAdvantagesRequest(
        samples=[_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm", return_diagnostics=True
    )
    legacy_req = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=_LegacyRequest)
    assert legacy_req.algorithm == "arm" and legacy_req.is_terminal is True
    resp = ComputeAdvantagesResponse(paired_samples=[_paired()], diagnostics=[], weight_step=3)
    legacy_resp = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=_LegacyResponse)
    assert legacy_resp.paired_samples[0].llm_sample.advantages == [0.5, 0.5, 0.5]


def test_request_with_flag_round_trips():
    req = ComputeAdvantagesRequest(
        samples=[_sample()], episodic_reward=1.0, is_terminal=True, algorithm="arm", return_diagnostics=True
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=ComputeAdvantagesRequest)
    assert decoded.return_diagnostics is True


def test_response_without_diagnostics_is_legacy_array_plus_nils():
    legacy = _LegacyResponse(paired_samples=[_paired()])
    new = ComputeAdvantagesResponse(paired_samples=[_paired()])
    assert _as_array(new) == _as_array(legacy) + [None, None]
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(legacy), type=ComputeAdvantagesResponse)
    assert decoded.diagnostics is None
    assert decoded.weight_step is None
    assert decoded.paired_samples[0].llm_sample.advantages == [0.5, 0.5, 0.5]


def test_diagnostics_round_trip():
    diag = DecisionPointDiagnostics(
        sample_idx=0,
        response_start=4,
        q_plus=[0.1, 0.25, -0.3],
        v=0.2,
        v_target=0.19,
        pi_rm=[0.0, 1.0, 0.0],
        adv_star=1.2345,
        g_k=1.0,
        q_plus_target_star=1.05,
    )
    resp = ComputeAdvantagesResponse(paired_samples=[_paired()], diagnostics=[diag], weight_step=7)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=ComputeAdvantagesResponse)
    assert decoded.weight_step == 7
    assert decoded.diagnostics is not None and len(decoded.diagnostics) == 1
    d = decoded.diagnostics[0]
    assert d.q_plus == [0.1, 0.25, -0.3]
    assert d.pi_rm == [0.0, 1.0, 0.0]
    assert d.adv_star == 1.2345
    assert d.g_k == 1.0
    assert d.q_plus_target_star == 1.05
    assert d.response_start == 4


def test_response_with_empty_diagnostics_is_distinguishable_from_none():
    # PPO-with-flag returns an empty list, not None (Phase 2 contract)
    resp = ComputeAdvantagesResponse(paired_samples=[_paired()], diagnostics=[], weight_step=0)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=ComputeAdvantagesResponse)
    assert decoded.diagnostics == []
    assert decoded.weight_step == 0
