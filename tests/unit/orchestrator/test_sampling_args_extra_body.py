"""Regression: user-set vLLM keys in ``SamplingConfig.extra_body`` must survive
``get_sampling_args``.

``get_sampling_args`` pins vLLM's neutral sampling defaults (``top_k=-1``,
``min_p=0.0``) so that, absent an explicit request, vLLM samples from the
full softmax and the model's ``generation_config.json`` defaults do not leak
in. Until the fix these were unconditional assignments made *after* copying
``sampling_config.extra_body``, so a run config's
``[orchestrator.sampling.extra_body] min_p = 0.1`` was silently replaced by
``0.0`` and never reached vLLM. The training ``SamplingConfig`` has no
top-level ``min_p`` / ``top_k`` field (only ``EvalSamplingConfig`` does), so
``extra_body`` was the only way to set them -- and it did not work.

Fix choice: ``extra_body.setdefault(...)`` for ``top_k`` / ``min_p`` -- the
neutral defaults apply only when the config did not set the key. Chosen over
adding top-level ``min_p`` / ``top_k`` fields to ``SamplingConfig`` (the
``EvalSamplingConfig`` shape) because it is a two-line change, every existing
run config stays valid and correct as written, no new config surface has to
be documented, and the behaviour for configs that never set these keys is
byte-identical. ``return_token_ids`` stays pinned to ``True``: the
orchestrator needs token IDs on every response, so it is not user-overridable.
"""

from __future__ import annotations

from prime_rl.configs.orchestrator import SamplingConfig
from prime_rl.orchestrator.utils import get_sampling_args


def test_extra_body_min_p_reaches_vllm():
    """A run config's ``extra_body.min_p`` must be forwarded unchanged."""
    config = SamplingConfig(max_tokens=128, extra_body={"min_p": 0.1})
    args = get_sampling_args(config, temperature=0.8, is_vllm=True)
    assert args["extra_body"]["min_p"] == 0.1


def test_extra_body_top_k_reaches_vllm():
    """Same contract for ``extra_body.top_k``."""
    config = SamplingConfig(max_tokens=128, extra_body={"top_k": 20})
    args = get_sampling_args(config, temperature=0.8, is_vllm=True)
    assert args["extra_body"]["top_k"] == 20


def test_vllm_neutral_defaults_when_unset():
    """Configs that do not set the keys keep today's behaviour exactly:
    ``top_k=-1``, ``min_p=0.0``, ``return_token_ids=True``."""
    config = SamplingConfig(max_tokens=128)
    args = get_sampling_args(config, temperature=0.8, is_vllm=True)
    extra = args["extra_body"]
    assert extra["top_k"] == -1
    assert extra["min_p"] == 0.0
    assert extra["return_token_ids"] is True


def test_return_token_ids_stays_pinned():
    """``return_token_ids`` is an orchestrator requirement, not a user knob."""
    config = SamplingConfig(max_tokens=128, extra_body={"return_token_ids": False})
    args = get_sampling_args(config, temperature=0.8, is_vllm=True)
    assert args["extra_body"]["return_token_ids"] is True


def test_non_vllm_path_leaves_extra_body_alone():
    """``is_vllm=False`` adds none of the vLLM-only keys and keeps user keys."""
    config = SamplingConfig(max_tokens=128, extra_body={"min_p": 0.1})
    args = get_sampling_args(config, temperature=0.8, is_vllm=False)
    extra = args["extra_body"]
    assert extra == {"min_p": 0.1}


def test_config_extra_body_not_mutated():
    """``get_sampling_args`` works on a copy; the config object is untouched."""
    config = SamplingConfig(max_tokens=128, extra_body={"min_p": 0.1})
    get_sampling_args(config, temperature=0.8, is_vllm=True)
    assert config.extra_body == {"min_p": 0.1}
