"""Regenerate GRPO golden-master fixture .pt files from the JSON inputs.

Usage:
    uv run python -m tests.unit.orchestrator.fixtures.regenerate_golden_master

This is a deliberate commit-time action, not a test-time one. It is invoked only
when an intentional GRPO change has shifted the reference outputs; the diff
summary it prints is meant to be inspected and pasted into the commit message
so the regeneration is traceable through history.

Fixtures regenerated:
    - grpo_golden_master.json -> grpo_golden_master.pt (length_shaping_alpha=None)
    - grpo_golden_master_gr3.json -> grpo_golden_master_gr3.pt (length_shaping_alpha=0.33)

Each .pt file holds a torch.Tensor of shape [num_problems, samples_per_problem]
(float32) holding the expected `advantages` output of `compute_advantages` for
the inputs in the matching .json file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import compute_advantages

FIXTURES_DIR = Path(__file__).parent

# (json basename, length_shaping_alpha)
FIXTURES: list[tuple[str, float | None]] = [
    ("grpo_golden_master", None),
    ("grpo_golden_master_gr3", 0.33),
]


def regenerate_one(
    name: str,
    length_shaping_alpha: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Read inputs from <name>.json, compute advantages, write to <name>.pt.

    Returns (new_tensor, old_tensor_or_None) for diff reporting.
    """
    json_path = FIXTURES_DIR / f"{name}.json"
    pt_path = FIXTURES_DIR / f"{name}.pt"

    inputs = json.loads(json_path.read_text())
    samples_per_problem: int = inputs["samples_per_problem"]
    rewards_flat = [r for row in inputs["rewards"] for r in row]
    lengths_flat = [length for row in inputs["completion_lengths"] for length in row]

    config = DefaultAdvantageConfig(length_shaping_alpha=length_shaping_alpha)
    flat = compute_advantages(rewards_flat, lengths_flat, samples_per_problem, config)
    new_tensor = torch.tensor(flat, dtype=torch.float32).view(-1, samples_per_problem)

    old_tensor: torch.Tensor | None = None
    if pt_path.exists():
        old_tensor = torch.load(pt_path, weights_only=True)

    torch.save(new_tensor, pt_path)
    return new_tensor, old_tensor


def main() -> None:
    for name, alpha in FIXTURES:
        new, old = regenerate_one(name, alpha)
        print(f"=== {name} (length_shaping_alpha={alpha}) ===")
        print(f"  shape: {tuple(new.shape)}, dtype: {new.dtype}")
        print(f"  output:\n{new}")
        if old is None:
            print("  (no previous .pt file; this is the initial generation)")
        else:
            diff = (new - old).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  vs previous: max |diff|={max_diff:.3e}, mean |diff|={mean_diff:.3e}")
            if max_diff == 0.0:
                print("  (byte-identical to previous .pt file)")


if __name__ == "__main__":
    main()
