"""Average the weights of several value_state.pt files into one (plan
bulletproof-warm-start, Round 0 step 0e: "stop early and average").

The inputs are checkpoints of ONE training run (the same parameter names and
shapes), close to each other in weight space. Every trainable tensor is averaged
in float32 and written back in the first input's dtype, in the format the
Advantage Server and Trainer load as a warm start. Whether the average scores
better than its members is read on the gold test set, not assumed.

    python -m prime_rl.value_warmstart.average_states @ average.toml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
from pydantic import Field

from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger


class AverageStatesConfig(BaseConfig):
    """Average value_state.pt files of one run."""

    inputs: Annotated[list[Path], Field(min_length=1, description="value_state.pt files to average.")]
    out: Annotated[Path, Field(description="Output value_state.pt (parent folders are created).")]
    log: Annotated[LogConfig, Field(description="Logging configuration.")] = LogConfig()


def average_trainable(payloads: list[dict]) -> dict:
    """The mean of the payloads' trainable_state_dict tensors.

    Raises when the payloads do not hold the same parameter names and shapes:
    averaging networks of different runs or shapes is never what is meant."""
    if not payloads:
        raise ValueError("nothing to average")
    first = payloads[0]["trainable_state_dict"]
    names = list(first)
    for i, p in enumerate(payloads[1:], start=1):
        sd = p["trainable_state_dict"]
        if set(sd) != set(names):
            missing = sorted(set(names) ^ set(sd))[:3]
            raise ValueError(f"input {i} does not hold the same parameters as input 0 (e.g. {missing})")
        for n in names:
            if sd[n].shape != first[n].shape:
                raise ValueError(f"input {i}: {n} has shape {tuple(sd[n].shape)}, input 0 has {tuple(first[n].shape)}")
    mean: dict[str, torch.Tensor] = {}
    for n in names:
        acc = torch.zeros_like(first[n], dtype=torch.float32)
        for p in payloads:
            acc += p["trainable_state_dict"][n].to(torch.float32)
        mean[n] = (acc / len(payloads)).to(first[n].dtype)
    steps = [int(p["step"]) for p in payloads if "step" in p]
    return {
        "step": max(steps) if steps else 0,
        "trainable_state_dict": mean,
        "averaged_from_steps": steps,
    }


def average_states(config: AverageStatesConfig) -> dict:
    logger = setup_logger(config.log.level)
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in config.inputs]
    out = average_trainable(payloads)
    out["averaged_from"] = [str(p) for p in config.inputs]
    Path(config.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, config.out)
    logger.success(
        f"averaged {len(payloads)} value state(s) (steps {out['averaged_from_steps']}), "
        f"{len(out['trainable_state_dict'])} tensors -> {config.out}"
    )
    return out


def main() -> None:
    """Entry point: `python -m prime_rl.value_warmstart.average_states @ average.toml`."""
    average_states(cli(AverageStatesConfig))


if __name__ == "__main__":
    main()
