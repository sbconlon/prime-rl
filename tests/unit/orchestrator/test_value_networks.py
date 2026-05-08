"""Phase 4 -- ValueNetworkBackbone tests.

Eight test categories from phase-04-encoder-lora-architecture.md section 10:
    1. Construction & parameter accounting
    2. Gradient masks
    3. Forward-pass shapes
    4. Zero-initialization
    5. Polyak update mechanics
    6. Determinism

Most tests use a tiny Qwen2 config (no model download, runs fast on CPU)
to verify architectural correctness. The two tests that assert ~1.5B
parameter counts use the real Qwen/Qwen2.5-1.5B-Instruct and are gated
behind pytest.mark.gpu + pytest.mark.slow.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.value_networks import (
    N_ADAPTERS,
    Q_PLUS_SLOT,
    V_SLOT,
    V_TARGET_SLOT,
    ValueHead,
    ValueNetworkBackbone,
    ValueNetworkConfig,
)
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_qwen2_config() -> Qwen2Config:
    """A 2-layer Qwen2 config small enough to construct in <1s on CPU.

    Hidden dims are kept divisible by 8 to satisfy MultiLoRALinear's
    grouped_mm shape constraint.
    """
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_backbone() -> ValueNetworkBackbone:
    torch.manual_seed(0)
    base_model = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    return ValueNetworkBackbone(base_model, lora_config=lora, polyak_tau=0.005)


# ---------------------------------------------------------------------------
# 1. Construction and parameter accounting
# ---------------------------------------------------------------------------


def test_construct_with_tiny_config():
    """ValueNetworkBackbone constructs successfully on a tiny Qwen2 model."""
    base_model = Qwen2Model(_tiny_qwen2_config())
    lora = LoRAConfig(rank=8, alpha=16.0, dropout=0.0)
    backbone = ValueNetworkBackbone(base_model, lora_config=lora, polyak_tau=0.005)
    assert isinstance(backbone, torch.nn.Module)
    # Three value heads exist.
    assert isinstance(backbone.v_head, ValueHead)
    assert isinstance(backbone.q_plus_head, ValueHead)
    assert isinstance(backbone.v_target_head, ValueHead)


@pytest.mark.gpu
@pytest.mark.slow
def test_construct_with_default_config_real_qwen():
    """Phase doc commitment: construct against the real Qwen2.5-1.5B-Instruct.

    Marked slow + gpu because it downloads ~3 GB and uses ~6 GB memory.
    """
    config = ValueNetworkConfig(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        lora=LoRAConfig(),
    )
    backbone = ValueNetworkBackbone.from_pretrained(config)
    assert isinstance(backbone, torch.nn.Module)
    # Approximate parameter accounting: ~1.5B base parameters.
    base_params = sum(
        p.numel()
        for n, p in backbone.named_parameters()
        if "lora_A" not in n and "lora_B" not in n and not n.startswith(
            ("v_head.", "q_plus_head.", "v_target_head.")
        )
    )
    assert 1.4e9 < base_params < 1.6e9, f"base_params {base_params} not ~1.5B"


def test_trainable_param_counts(tiny_backbone: ValueNetworkBackbone):
    """Only LoRA adapters and value heads are trainable; base is frozen."""
    trainable = [
        (n, p) for n, p in tiny_backbone.named_parameters() if p.requires_grad
    ]
    # Every trainable parameter must be either LoRA or a value head.
    for n, _ in trainable:
        is_lora = "lora_A" in n or "lora_B" in n
        is_head = (
            n.startswith("v_head.")
            or n.startswith("q_plus_head.")
            or n.startswith("v_target_head.")
        )
        assert is_lora or is_head, f"unexpected trainable parameter: {n}"
    # And every LoRA / head parameter must be trainable.
    for n, p in tiny_backbone.named_parameters():
        is_lora = "lora_A" in n or "lora_B" in n
        is_head = (
            n.startswith("v_head.")
            or n.startswith("q_plus_head.")
            or n.startswith("v_target_head.")
        )
        if is_lora or is_head:
            assert p.requires_grad, f"LoRA/head parameter {n} should require grad"


def test_lora_modules_count_matches_target_modules(tiny_backbone: ValueNetworkBackbone):
    """Phase doc DQ1: LoRA at every target module in every layer.

    Tiny config has 2 layers × 7 target modules (q/k/v/o/gate/up/down) = 14.
    """
    n_lora = sum(
        1 for m in tiny_backbone.base_model.modules() if isinstance(m, MultiLoRALinear)
    )
    assert n_lora == 14, f"expected 14 MultiLoRALinear modules, got {n_lora}"


def test_each_lora_module_has_three_adapters(tiny_backbone: ValueNetworkBackbone):
    """Per the architecture: V, Q+, V_target are three slots in each MultiLoRALinear."""
    for m in tiny_backbone.base_model.modules():
        if isinstance(m, MultiLoRALinear):
            assert m.n_adapters == N_ADAPTERS == 3
            assert len(m.lora_A) == 3
            assert len(m.lora_B) == 3


# ---------------------------------------------------------------------------
# 2. Gradient masks
# ---------------------------------------------------------------------------


def test_base_weights_frozen(tiny_backbone: ValueNetworkBackbone):
    """Base model parameters (non-LoRA, non-head) all have requires_grad=False."""
    for name, p in tiny_backbone.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = name.startswith(("v_head.", "q_plus_head.", "v_target_head."))
        if not (is_lora or is_head):
            assert not p.requires_grad, f"base parameter '{name}' should be frozen"


def test_lora_adapters_trainable(tiny_backbone: ValueNetworkBackbone):
    """All lora_A / lora_B parameters have requires_grad=True."""
    found_any = False
    for name, p in tiny_backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            found_any = True
            assert p.requires_grad, f"LoRA parameter '{name}' must be trainable"
    assert found_any, "no LoRA parameters found"


def test_value_heads_trainable(tiny_backbone: ValueNetworkBackbone):
    """All v_head / q_plus_head / v_target_head parameters require grad."""
    head_prefixes = ("v_head.", "q_plus_head.", "v_target_head.")
    found_per_head = {p: 0 for p in head_prefixes}
    for name, p in tiny_backbone.named_parameters():
        for prefix in head_prefixes:
            if name.startswith(prefix):
                assert p.requires_grad, f"head parameter '{name}' must be trainable"
                found_per_head[prefix] += 1
    # Each ValueHead has 2 parameters (linear weight + bias).
    for prefix, count in found_per_head.items():
        assert count == 2, f"expected 2 params under {prefix}, found {count}"


def test_freeze_base_weights_assertion_passes(tiny_backbone: ValueNetworkBackbone):
    """The freeze_base_weights() utility passes after correct __init__ setup."""
    tiny_backbone.freeze_base_weights()  # raises if anything is wrong


# ---------------------------------------------------------------------------
# 3. Forward pass shapes
# ---------------------------------------------------------------------------


def test_forward_v_returns_scalar_per_batch(tiny_backbone: ValueNetworkBackbone):
    """forward_v returns shape [batch] (one scalar per batch element)."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out = tiny_backbone.forward_v(input_ids)
    assert out.shape == (1,)


def test_forward_q_plus_returns_scalar_per_batch(tiny_backbone: ValueNetworkBackbone):
    """forward_q_plus returns shape [batch]."""
    obs = torch.tensor([[1, 2, 3, 4]])
    out = tiny_backbone.forward_q_plus(obs, action_id=5)
    assert out.shape == (1,)


def test_forward_q_plus_appends_action(tiny_backbone: ValueNetworkBackbone):
    """forward_q_plus runs the base on the [obs, action] concatenation.

    Patch the base model's forward to capture the input_ids it receives.
    Verify the captured tensor equals [obs, action_id].
    """
    obs = torch.tensor([[10, 20, 30, 40]])
    action_id = 99

    captured: dict[str, torch.Tensor] = {}
    real_forward = tiny_backbone.base_model.forward

    def spy_forward(input_ids, *args, **kwargs):
        captured["input_ids"] = input_ids.clone()
        return real_forward(input_ids, *args, **kwargs)

    tiny_backbone.base_model.forward = spy_forward
    try:
        tiny_backbone.forward_q_plus(obs, action_id=action_id)
    finally:
        tiny_backbone.base_model.forward = real_forward

    expected = torch.tensor([[10, 20, 30, 40, 99]])
    assert torch.equal(captured["input_ids"], expected), (
        f"forward_q_plus passed unexpected input_ids: {captured['input_ids']}"
    )


def test_forward_v_target_returns_scalar(tiny_backbone: ValueNetworkBackbone):
    """forward_v_target returns shape [batch]."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out = tiny_backbone.forward_v_target(input_ids)
    assert out.shape == (1,)


def test_forward_batched(tiny_backbone: ValueNetworkBackbone):
    """forward_v handles batched input correctly (returns [batch] outputs)."""
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = tiny_backbone.forward_v(input_ids)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# 4. Zero-initialization
# ---------------------------------------------------------------------------


def test_v_outputs_zero_at_init(tiny_backbone: ValueNetworkBackbone):
    """Fresh ValueNetworkBackbone outputs exactly 0 from forward_v.

    LoRA's lora_B is zero-initialized so the adapter contribution is zero;
    the value head is also zero-initialized. End-to-end output is exactly 0.
    """
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out = tiny_backbone.forward_v(input_ids)
    assert out.abs().max().item() < 1e-6


def test_q_plus_outputs_zero_at_init(tiny_backbone: ValueNetworkBackbone):
    """Fresh ValueNetworkBackbone outputs exactly 0 from forward_q_plus."""
    obs = torch.tensor([[1, 2, 3, 4]])
    out = tiny_backbone.forward_q_plus(obs, action_id=5)
    assert out.abs().max().item() < 1e-6


def test_v_target_outputs_zero_at_init(tiny_backbone: ValueNetworkBackbone):
    """Fresh ValueNetworkBackbone outputs exactly 0 from forward_v_target."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out = tiny_backbone.forward_v_target(input_ids)
    assert out.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 5. Polyak update mechanics
# ---------------------------------------------------------------------------


def _set_v_and_v_target_lora(
    backbone: ValueNetworkBackbone, v_value: float, v_target_value: float
) -> None:
    """Helper: set every LoRA module's V slot to v_value, V_target slot to v_target_value."""
    with torch.no_grad():
        for module in backbone.base_model.modules():
            if isinstance(module, MultiLoRALinear):
                module.lora_A[V_SLOT].fill_(v_value)
                module.lora_B[V_SLOT].fill_(v_value)
                module.lora_A[V_TARGET_SLOT].fill_(v_target_value)
                module.lora_B[V_TARGET_SLOT].fill_(v_target_value)


def test_polyak_update_with_tau_zero_preserves_v_target(
    tiny_backbone: ValueNetworkBackbone,
):
    """tau=0 -> V_target unchanged (V's contribution weighted by 0)."""
    _set_v_and_v_target_lora(tiny_backbone, v_value=1.0, v_target_value=0.0)
    tiny_backbone.polyak_update_v_target(tau=0.0)
    for module in tiny_backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            assert torch.allclose(
                module.lora_A[V_TARGET_SLOT],
                torch.zeros_like(module.lora_A[V_TARGET_SLOT]),
            )
            assert torch.allclose(
                module.lora_B[V_TARGET_SLOT],
                torch.zeros_like(module.lora_B[V_TARGET_SLOT]),
            )


def test_polyak_update_with_tau_one_overwrites_v_target(
    tiny_backbone: ValueNetworkBackbone,
):
    """tau=1 -> V_target equals V exactly (V_target's prior contribution weighted by 0)."""
    _set_v_and_v_target_lora(tiny_backbone, v_value=1.0, v_target_value=0.0)
    tiny_backbone.polyak_update_v_target(tau=1.0)
    for module in tiny_backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            assert torch.allclose(
                module.lora_A[V_TARGET_SLOT], module.lora_A[V_SLOT]
            )
            assert torch.allclose(
                module.lora_B[V_TARGET_SLOT], module.lora_B[V_SLOT]
            )


def test_polyak_update_with_tau_half_averages(tiny_backbone: ValueNetworkBackbone):
    """tau=0.5 -> V_target = 0.5 * V + 0.5 * V_target_old (with V=1, V_target_old=0 -> 0.5)."""
    _set_v_and_v_target_lora(tiny_backbone, v_value=1.0, v_target_value=0.0)
    tiny_backbone.polyak_update_v_target(tau=0.5)
    for module in tiny_backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            assert torch.allclose(
                module.lora_A[V_TARGET_SLOT],
                torch.full_like(module.lora_A[V_TARGET_SLOT], 0.5),
            )
            assert torch.allclose(
                module.lora_B[V_TARGET_SLOT],
                torch.full_like(module.lora_B[V_TARGET_SLOT], 0.5),
            )


def test_polyak_update_includes_value_head(tiny_backbone: ValueNetworkBackbone):
    """v_target_head's parameters are also Polyak-averaged from v_head's, not just LoRA."""
    with torch.no_grad():
        tiny_backbone.v_head.linear.weight.fill_(1.0)
        tiny_backbone.v_head.linear.bias.fill_(2.0)
        tiny_backbone.v_target_head.linear.weight.fill_(0.0)
        tiny_backbone.v_target_head.linear.bias.fill_(0.0)
    tiny_backbone.polyak_update_v_target(tau=0.5)
    assert torch.allclose(
        tiny_backbone.v_target_head.linear.weight,
        torch.full_like(tiny_backbone.v_target_head.linear.weight, 0.5),
    )
    assert torch.allclose(
        tiny_backbone.v_target_head.linear.bias,
        torch.full_like(tiny_backbone.v_target_head.linear.bias, 1.0),  # 0.5 * 2.0
    )


def test_polyak_update_default_tau_uses_config(tiny_backbone: ValueNetworkBackbone):
    """Calling polyak_update_v_target() with no argument uses self.polyak_tau."""
    _set_v_and_v_target_lora(tiny_backbone, v_value=1.0, v_target_value=0.0)
    # tiny_backbone.polyak_tau == 0.005 (set in fixture).
    tiny_backbone.polyak_update_v_target()
    for module in tiny_backbone.base_model.modules():
        if isinstance(module, MultiLoRALinear):
            # After one step at tau=0.005: V_target = 0.005 * 1 + 0.995 * 0 = 0.005
            assert torch.allclose(
                module.lora_B[V_TARGET_SLOT],
                torch.full_like(module.lora_B[V_TARGET_SLOT], 0.005),
                atol=1e-7,
            )


def test_polyak_update_does_not_change_v(tiny_backbone: ValueNetworkBackbone):
    """V's adapter slot is the source, never the destination -- it stays unchanged."""
    _set_v_and_v_target_lora(tiny_backbone, v_value=1.0, v_target_value=0.0)
    # Snapshot V's slot before update.
    snapshot = {}
    for n, m in tiny_backbone.base_model.named_modules():
        if isinstance(m, MultiLoRALinear):
            snapshot[n] = (m.lora_A[V_SLOT].clone(), m.lora_B[V_SLOT].clone())
    tiny_backbone.polyak_update_v_target(tau=0.5)
    for n, m in tiny_backbone.base_model.named_modules():
        if isinstance(m, MultiLoRALinear):
            a_before, b_before = snapshot[n]
            assert torch.allclose(m.lora_A[V_SLOT], a_before)
            assert torch.allclose(m.lora_B[V_SLOT], b_before)


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_forward_v_deterministic(tiny_backbone: ValueNetworkBackbone):
    """forward_v on the same input twice returns bit-identical output."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    out_a = tiny_backbone.forward_v(input_ids)
    out_b = tiny_backbone.forward_v(input_ids)
    assert torch.equal(out_a, out_b)


def test_forward_q_plus_deterministic(tiny_backbone: ValueNetworkBackbone):
    """forward_q_plus is deterministic across two identical calls."""
    obs = torch.tensor([[1, 2, 3, 4]])
    out_a = tiny_backbone.forward_q_plus(obs, action_id=5)
    out_b = tiny_backbone.forward_q_plus(obs, action_id=5)
    assert torch.equal(out_a, out_b)


# ===========================================================================
# Phase 6.5 -- optimized-vs-naive correctness oracle tests
# ===========================================================================
# The naive forward methods (Phase 4) serve as the correctness oracle for the
# optimized methods (Phase 6.5). For each optimized method, the test asserts
# the optimized output matches the per-position naive output within float
# tolerance.
# ---------------------------------------------------------------------------


def _backbone_with_distinct_adapters_and_heads() -> ValueNetworkBackbone:
    """Build a tiny ValueNetworkBackbone with non-zero LoRA + non-zero value
    heads. Production zero-init makes iteration-0 outputs trivially zero;
    these tests exercise the non-trivial path.
    """
    import torch
    from transformers import Qwen2Config, Qwen2Model

    from prime_rl.configs.trainer import LoRAConfig
    from prime_rl.orchestrator.value_networks import (
        Q_PLUS_SLOT,
        V_SLOT,
        V_TARGET_SLOT,
        ValueNetworkBackbone,
    )
    from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    backbone.eval()
    # Set distinct LoRA values per slot.
    with torch.no_grad():
        for module in backbone.base_model.modules():
            if isinstance(module, MultiLoRALinear):
                module.lora_A[V_SLOT].fill_(0.4)
                module.lora_B[V_SLOT].fill_(0.4)
                module.lora_A[Q_PLUS_SLOT].fill_(0.6)
                module.lora_B[Q_PLUS_SLOT].fill_(0.6)
                module.lora_A[V_TARGET_SLOT].fill_(0.5)
                module.lora_B[V_TARGET_SLOT].fill_(0.5)
        # Break head zero-init so non-trivial dynamics flow through.
        for head in (backbone.v_head, backbone.q_plus_head, backbone.v_target_head):
            torch.nn.init.normal_(head.linear.weight, mean=0.0, std=0.02)
    return backbone


def test_forward_v_all_positions_matches_per_position_naive():
    """Single all-positions forward gives the same V values as N naive
    forwards over progressive prefixes."""
    import torch

    backbone = _backbone_with_distinct_adapters_and_heads()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    seq_len = input_ids.shape[1]

    naive = [backbone.forward_v(input_ids[:, : k + 1]).item() for k in range(seq_len)]
    optimized, _ = backbone.forward_v_all_positions(input_ids)
    optimized_list = optimized[0].tolist()

    for k, (a, b) in enumerate(zip(naive, optimized_list)):
        assert abs(a - b) < 1e-4, f"position {k}: naive={a}, optimized={b}"


def test_forward_v_target_all_positions_matches_naive():
    """Same correctness oracle for V_target slot."""
    import torch

    backbone = _backbone_with_distinct_adapters_and_heads()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    seq_len = input_ids.shape[1]

    naive = [backbone.forward_v_target(input_ids[:, : k + 1]).item() for k in range(seq_len)]
    optimized, _ = backbone.forward_v_target_all_positions(input_ids)
    optimized_list = optimized[0].tolist()

    for k, (a, b) in enumerate(zip(naive, optimized_list)):
        assert abs(a - b) < 1e-4, f"position {k}: naive={a}, optimized={b}"


def test_forward_q_plus_sampled_all_positions_matches_naive():
    """For each position k, optimized Q+ at the sampled action equals naive
    forward_q_plus over the prefix-before-k with the sampled action appended.

    Specifically, the optimized method reads q_plus_head at position k of
    a forward over [prompt + completion]. That's equivalent to:
        prefix = prompt + completion[:k]
        sampled = completion[k]
        forward_q_plus(prefix, sampled)
    """
    import torch

    backbone = _backbone_with_distinct_adapters_and_heads()
    # Use a short input so naive iteration is fast.
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    seq_len = input_ids.shape[1]

    optimized, _ = backbone.forward_q_plus_sampled_all_positions(input_ids)
    optimized_list = optimized[0].tolist()

    # Naive: for each position k, append completion[k] to the prefix-before-k.
    # The prefix-before-k is input_ids[:, :k]; the sampled action at k is
    # input_ids[0, k].
    for k in range(seq_len):
        if k == 0:
            # Skip k=0: forward_q_plus needs a non-empty prefix.
            continue
        prefix = input_ids[:, :k]
        sampled = int(input_ids[0, k].item())
        naive_qpk = backbone.forward_q_plus(prefix, action_id=sampled).item()
        assert abs(naive_qpk - optimized_list[k]) < 1e-4, (
            f"position {k}: naive={naive_qpk}, optimized={optimized_list[k]}"
        )


def test_forward_q_plus_candidates_at_position_batched_matches_naive():
    """Batched K-candidate forward with cloned+cropped+tiled cache matches
    K naive forward_q_plus calls, each with a single candidate."""
    import torch

    from prime_rl.orchestrator.value_networks import ValueNetworkBackbone

    backbone = _backbone_with_distinct_adapters_and_heads()
    full_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # [1, 8]
    prefix_length = 5  # evaluate candidates at position 5 (after 5 tokens)
    candidates = [10, 20, 30, 40]
    K = len(candidates)

    # Naive: for each candidate, full forward_q_plus(prefix=full_input[:, :prefix_length], action_id=candidate).
    prefix = full_input[:, :prefix_length]
    naive = [backbone.forward_q_plus(prefix, action_id=c).item() for c in candidates]

    # Optimized: forward Q+ over full_input to get the cache, clone+crop to
    # `prefix_length`, tile to K, batched forward.
    _, q_cache = backbone.forward_q_plus_sampled_all_positions(full_input)
    cropped = ValueNetworkBackbone.clone_and_crop_cache(q_cache, prefix_length)
    candidates_tensor = torch.tensor(candidates, dtype=torch.long)
    optimized = backbone.forward_q_plus_candidates_at_position(
        cropped, candidates_tensor, prefix_length
    ).tolist()

    for i, (a, b) in enumerate(zip(naive, optimized)):
        assert abs(a - b) < 1e-3, f"candidate {i} ({candidates[i]}): naive={a}, optimized={b}"


def test_clone_and_crop_cache_preserves_original():
    """Clone + crop the cache; verify the original cache is unchanged
    (so callers can use the same cache at multiple positions)."""
    import torch

    from prime_rl.orchestrator.value_networks import ValueNetworkBackbone

    backbone = _backbone_with_distinct_adapters_and_heads()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    _, cache = backbone.forward_q_plus_sampled_all_positions(input_ids)

    original_keys_shape = tuple(cache.layers[0].keys.shape)
    cropped = ValueNetworkBackbone.clone_and_crop_cache(cache, 3)

    # Mutate the clone -- batch_repeat_interleave is in-place.
    cropped.batch_repeat_interleave(2)
    assert tuple(cropped.layers[0].keys.shape) == (2, 4, 3, 16)

    # Original is unchanged.
    assert tuple(cache.layers[0].keys.shape) == original_keys_shape


def test_optimized_q_plus_at_zero_init_matches_zero():
    """Sanity: at iteration-0 (zero-init adapters and heads),
    forward_q_plus_sampled_all_positions returns all zeros, matching the
    iteration-0 invariant for ARM's cold-start branch."""
    import torch
    from transformers import Qwen2Config, Qwen2Model

    from prime_rl.configs.trainer import LoRAConfig
    from prime_rl.orchestrator.value_networks import ValueNetworkBackbone

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    backbone = ValueNetworkBackbone(
        Qwen2Model(config),
        lora_config=LoRAConfig(rank=8, alpha=16.0, dropout=0.0),
        polyak_tau=0.005,
    )
    backbone.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    q_all, _ = backbone.forward_q_plus_sampled_all_positions(input_ids)
    assert q_all.abs().max().item() < 1e-7
