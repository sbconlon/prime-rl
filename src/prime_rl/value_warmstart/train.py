"""Offline warm-start training (Phase 3).

Loads ValueNetworkBackbone from the SFT base, runs SFT-style epochs over the
Phase-2 dataset using the shared V/Q+ regression legs (byte-identical to the
online Advantage Trainer), seeds V_target = V, and saves a weights-only
value_state.pt at step 0. See phase-03-training-pipeline.md.
"""

from __future__ import annotations

from typing import Iterator

import torch
from transformers import AutoModel, AutoTokenizer

from prime_rl.advantage_trainer._prof import set_request_id
from prime_rl.advantage_trainer.ckpt import setup_advantage_ckpt_manager
from prime_rl.advantage_trainer.shared import (
    setup_value_optimizer,
    value_regression_backward,
)
from prime_rl.configs.value_warmstart import WarmStartTrainConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.data import WarmStartDataset


def _shuffled_batches(n: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """Indices for one epoch: a fresh shuffle split into batches of `batch_size`."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def train(config: WarmStartTrainConfig) -> None:
    logger = setup_logger(config.log.level)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    logger.info(
        f"Loading value backbone from {config.model.base_model_name} "
        f"(device={device}, dtype={dtype})"
    )
    base = AutoModel.from_pretrained(config.model.base_model_name, dtype=dtype)
    backbone = ValueNetworkBackbone(
        base, lora_config=config.model.lora, polyak_tau=config.model.polyak_tau
    ).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
    optimizer = setup_value_optimizer(
        backbone, config.v_learning_rate, config.q_plus_learning_rate
    )

    dataset = WarmStartDataset(config.data.path)
    logger.info(f"Loaded {len(dataset)} warm-start samples from {config.data.path}")
    if len(dataset) == 0:
        raise ValueError(f"Warm-start dataset at {config.data.path} is empty.")

    backbone.train()
    step = 0
    for epoch in range(config.max_epochs):
        epoch_l_v = epoch_l_q = 0.0
        n_batches = 0
        for batch_idx in _shuffled_batches(
            len(dataset), config.data.batch_size, config.seed + epoch
        ):
            set_request_id(str(step))
            batch = [dataset[i] for i in batch_idx]
            optimizer.zero_grad()
            b_l_v = b_l_q = 0.0
            for chunk_start in range(0, len(batch), config.data.inner_batch_size):
                chunk = batch[chunk_start : chunk_start + config.data.inner_batch_size]
                chunk_scale = float(len(chunk)) / float(len(batch))
                l_v, l_q = value_regression_backward(
                    backbone, chunk, tokenizer, chunk_scale, device
                )
                b_l_v += l_v
                b_l_q += l_q
            optimizer.step()
            epoch_l_v += b_l_v
            epoch_l_q += b_l_q
            n_batches += 1
            step += 1
        logger.info(
            f"epoch={epoch} l_v={epoch_l_v / n_batches:.5f} "
            f"l_q={epoch_l_q / n_batches:.5f} steps={n_batches}"
        )

    # Seed V_target = V exactly, then save the weights-only value_state.pt at step 0
    # (the RL run resumes from *its* step 0; skip_optimizer keeps it weights-only).
    backbone.polyak_update_v_target(tau=1.0)
    ckpt_manager = setup_advantage_ckpt_manager(config.output_dir, config.ckpt)
    if ckpt_manager is None:
        raise ValueError("WarmStartTrainConfig.ckpt is required; it produces value_state.pt.")
    ckpt_manager.save(step=0, backbone=backbone, optimizer=optimizer)
    logger.success(f"Warm-start done. value_state.pt at {ckpt_manager.get_ckpt_path(0)}")


def main() -> None:
    """Entry point: `uv run warmstart-train @ warmstart-train.toml`."""
    train(cli(WarmStartTrainConfig))


if __name__ == "__main__":
    main()
