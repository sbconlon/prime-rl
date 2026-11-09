"""Offline warm-start training (Phase 3).

Loads ValueNetworkBackbone from the SFT base, runs SFT-style epochs over the
warm-start dataset using the shared V/Q+ regression legs (byte-identical to the
online Advantage Trainer), seeds V_target = V, and saves a weights-only
value_state.pt at step 0. See phase-03-training-pipeline.md.

Branched dataset (plan branched-dataset Phase 3): `data.kind = "branched"` reads
BranchRecords through BranchedDataset (pre-root prefix unlabelled, forced root
labels Q+ only), excludes held-out games, logs the losses split by provenance
(trunk vs branch episodes), and writes holdout.json + summary.json next to the
checkpoint.

Resume: after every epoch a full checkpoint (trainable weights + optimizer
state, step = number of completed epochs) is written under output_dir/resume,
keeping only the latest. With config.resume (default) a run that finds one
continues from it and trains the remaining epochs up to max_epochs, so a
walltime-killed job is simply resubmitted. The weights-only value_state.pt at
checkpoints/step_0 (what the RL run and the gate read) is unchanged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModel, AutoTokenizer

from prime_rl.advantage_trainer._prof import set_request_id
from prime_rl.advantage_trainer.ckpt import AdvantageCheckpointManager
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.advantage_trainer.shared import (
    setup_value_optimizer,
    value_regression_backward,
)
from prime_rl.configs.value_warmstart import WarmStartTrainConfig
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.value_warmstart.data import BranchedDataset, WarmStartDataset

PROVENANCES = ("trunk", "branch")


def _shuffled_batches(n: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """Indices for one epoch: a fresh shuffle split into batches of `batch_size`."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def _load_holdout_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with open(path) as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise ValueError(f"holdout_ids_path {path} must hold a JSON list of game ids.")
    return {str(i) for i in ids}


def _load_dataset(config: WarmStartTrainConfig):
    """The dataset for config.data.kind, plus its per-sample provenance list and
    the held-out game ids (empty for the single-action kind)."""
    if config.data.kind == "branched":
        ds = BranchedDataset(
            config.data.path,
            holdout_ids=_load_holdout_ids(config.data.holdout_ids_path),
            include_holdout=config.data.include_holdout,
            min_p_accept=config.data.min_p_accept,
        )
        return ds, list(ds.provenance), sorted(ds.holdout_game_ids)
    ds = WarmStartDataset(config.data.path)
    return ds, ["trunk"] * len(ds), []


def _group_by_provenance(
    chunk_idx: list[int], provenance: list[str]
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i in chunk_idx:
        groups[provenance[i]].append(i)
    return groups


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def train(
    config: WarmStartTrainConfig,
    *,
    backbone: ValueNetworkBackbone | None = None,
    tokenizer=None,
) -> dict:
    """Run the warm-start regression. `backbone` / `tokenizer` may be injected
    (tests: the tiny CPU backbone); otherwise they are built from config.model.
    Returns the final summary dict (also written to output_dir/summary.json)."""
    logger = setup_logger(config.log.level)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    if backbone is None:
        logger.info(
            f"Loading value backbone from {config.model.base_model_name} "
            f"(device={device}, dtype={dtype})"
        )
        base = AutoModel.from_pretrained(config.model.base_model_name, dtype=dtype)
        backbone = ValueNetworkBackbone(
            base, lora_config=config.model.lora, polyak_tau=config.model.polyak_tau
        ).to(device=device, dtype=dtype)
    else:
        device = next(backbone.parameters()).device
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
    optimizer = setup_value_optimizer(
        backbone, config.v_learning_rate, config.q_plus_learning_rate
    )

    dataset, provenance, holdout_ids = _load_dataset(config)
    logger.info(
        f"Loaded {len(dataset)} {config.data.kind} samples from {config.data.path} "
        f"(trunk={provenance.count('trunk')} branch={provenance.count('branch')} "
        f"holdout_games={len(holdout_ids)})"
    )
    if len(dataset) == 0:
        raise ValueError(f"Warm-start dataset at {config.data.path} is empty.")
    _write_json(Path(config.output_dir) / "holdout.json", {"game_ids": holdout_ids})

    # Save AFTER EACH EPOCH (overwriting step_0), not just at the end: over 12k+ samples
    # this can exceed the walltime, and an end-only save is all-or-nothing (a kill loses
    # everything). Per-epoch save keeps the latest trained net; watch the l_v/l_q plateau
    # and scancel when converged. Create the ckpt manager up front.
    ckpt_manager = AdvantageCheckpointManager(config.output_dir, CheckpointConfig(skip_optimizer=True))
    # Resume checkpoints: weights + optimizer, step = epochs completed, latest only.
    resume_manager = AdvantageCheckpointManager(
        Path(config.output_dir) / "resume", CheckpointConfig(skip_optimizer=False, keep_last=1)
    )
    start_epoch = 0
    if config.resume and resume_manager.ckpt_steps:
        last = resume_manager.ckpt_steps[-1]
        start_epoch = resume_manager.load(last, backbone, optimizer)
        logger.info(
            f"Resuming: {start_epoch} of {config.max_epochs} epochs already completed "
            f"({resume_manager.get_ckpt_path(last)})"
        )
    backbone.train()
    n_batches_per_epoch = -(-len(dataset) // config.data.batch_size)
    step = start_epoch * n_batches_per_epoch
    summary: dict = {
        "dataset_kind": config.data.kind,
        "dataset_path": str(config.data.path),
        "n_samples": len(dataset),
        "n_samples_by_provenance": {p: provenance.count(p) for p in PROVENANCES},
        "holdout_game_ids": holdout_ids,
        "epochs": [],
    }
    if config.data.kind == "branched":
        summary["n_root_targets"] = dataset.n_root_targets
        summary["n_dropped_empty_samples"] = dataset.n_dropped_empty
    summary_path = Path(config.output_dir) / "summary.json"
    if start_epoch > 0 and summary_path.exists():
        # keep the completed epochs' records so summary.json stays cumulative
        with open(summary_path) as f:
            previous = json.load(f)
        summary["epochs"] = list(previous.get("epochs", []))[:start_epoch]
        if summary["epochs"]:
            summary["final"] = summary["epochs"][-1]
    if start_epoch >= config.max_epochs:
        logger.info(f"Nothing to do: {start_epoch} epochs completed, max_epochs={config.max_epochs}")
        _write_json(summary_path, summary)
        return summary
    for epoch in range(start_epoch, config.max_epochs):
        epoch_l_v = epoch_l_q = 0.0
        by_prov = {p: {"l_v": 0.0, "l_q": 0.0, "n_samples": 0} for p in PROVENANCES}
        n_batches = 0
        for batch_idx in _shuffled_batches(
            len(dataset), config.data.batch_size, config.seed + epoch
        ):
            set_request_id(str(step))
            optimizer.zero_grad()
            b_l_v = b_l_q = 0.0
            for chunk_start in range(0, len(batch_idx), config.data.inner_batch_size):
                chunk_idx = batch_idx[chunk_start : chunk_start + config.data.inner_batch_size]
                # One legs call per provenance group inside the chunk: the summed
                # loss is the same as one call on the whole chunk (per-sample-equal
                # weights x chunk_scale), and the split is observable.
                for prov, idx in _group_by_provenance(chunk_idx, provenance).items():
                    group = [dataset[i] for i in idx]
                    chunk_scale = float(len(group)) / float(len(batch_idx))
                    l_v, l_q = value_regression_backward(
                        backbone, group, tokenizer, chunk_scale, device
                    )
                    b_l_v += l_v
                    b_l_q += l_q
                    by_prov[prov]["l_v"] += l_v
                    by_prov[prov]["l_q"] += l_q
                    by_prov[prov]["n_samples"] += len(group)
            optimizer.step()
            epoch_l_v += b_l_v
            epoch_l_q += b_l_q
            n_batches += 1
            step += 1
        # Seed V_target = V exactly and save the weights-only value_state.pt at step 0
        # (overwrites each epoch; the RL run resumes from *its* step 0).
        backbone.polyak_update_v_target(tau=1.0)
        ckpt_manager.save(step=0, backbone=backbone, optimizer=optimizer)
        resume_manager.save(step=epoch + 1, backbone=backbone, optimizer=optimizer)
        resume_manager.maybe_clean()
        epoch_summary = {
            "epoch": epoch,
            "steps": n_batches,
            "l_v": epoch_l_v / n_batches,
            "l_q": epoch_l_q / n_batches,
            # Per-provenance contributions to the batch-mean loss (they sum to l_v / l_q).
            "by_provenance": {
                p: {
                    "l_v": by_prov[p]["l_v"] / n_batches,
                    "l_q": by_prov[p]["l_q"] / n_batches,
                    "n_samples": by_prov[p]["n_samples"],
                }
                for p in PROVENANCES
            },
        }
        summary["epochs"].append(epoch_summary)
        summary["final"] = epoch_summary
        _write_json(summary_path, summary)
        split = " ".join(
            f"warmstart/l_v/{p}={epoch_summary['by_provenance'][p]['l_v']:.5f} "
            f"warmstart/l_q/{p}={epoch_summary['by_provenance'][p]['l_q']:.5f} "
            f"n/{p}={epoch_summary['by_provenance'][p]['n_samples']}"
            for p in PROVENANCES
            if epoch_summary["by_provenance"][p]["n_samples"] > 0
        )
        logger.success(
            f"epoch={epoch} l_v={epoch_summary['l_v']:.5f} "
            f"l_q={epoch_summary['l_q']:.5f} steps={n_batches} | {split} "
            f"-> saved value_state.pt (+ resume step {epoch + 1}/{config.max_epochs})"
        )
    return summary


def main() -> None:
    """Entry point: `uv run warmstart-train @ warmstart-train.toml`."""
    train(cli(WarmStartTrainConfig))


if __name__ == "__main__":
    main()
