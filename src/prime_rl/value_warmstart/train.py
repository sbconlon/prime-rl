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

Resume: every ckpt_every_steps optimizer steps and at each epoch end a full
checkpoint (trainable weights + optimizer state, step = global optimizer step)
is written under output_dir/resume, keeping only the latest, with the running
epoch's loss sums in resume/progress.json. With config.resume (default) a run
that finds one continues at the exact batch (the per-epoch shuffle depends only
on seed + epoch) and trains up to max_epochs in total, so a walltime-killed job
is simply resubmitted. The weights-only value_state.pt at checkpoints/step_0
(what the RL run and the gate read) is refreshed at every checkpoint.
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

    # Checkpointing. Two artifacts, both written at every checkpoint (every
    # config.ckpt_every_steps optimizer steps and at each epoch end):
    #   - checkpoints/step_0/trainer/value_state.pt: weights only, V_target = V. This
    #     is what the RL run, the gate and contrast_eval read. Overwritten each time.
    #   - resume/checkpoints/step_<global step>/: weights + optimizer, latest only, and
    #     resume/progress.json with the running epoch's loss sums.
    # A walltime-killed job resubmitted with the same command continues at the exact
    # batch: the per-epoch shuffle depends only on (seed, epoch), so skipping the
    # first k batches of the interrupted epoch reproduces the rest of its order.
    ckpt_manager = AdvantageCheckpointManager(config.output_dir, CheckpointConfig(skip_optimizer=True))
    resume_dir = Path(config.output_dir) / "resume"
    resume_manager = AdvantageCheckpointManager(
        resume_dir, CheckpointConfig(skip_optimizer=False, keep_last=config.resume_keep_last)
    )
    progress_path = resume_dir / "progress.json"
    n_batches_per_epoch = -(-len(dataset) // config.data.batch_size)
    total_steps = config.max_epochs * n_batches_per_epoch
    shape = {"n_samples": len(dataset), "batch_size": config.data.batch_size, "seed": config.seed}

    global_step = 0
    progress: dict = {}
    if config.resume and resume_manager.ckpt_steps:
        last = resume_manager.ckpt_steps[-1]
        if progress_path.exists():
            with open(progress_path) as f:
                progress = json.load(f)
        if progress and {k: progress.get(k) for k in shape} != shape:
            raise ValueError(
                f"Cannot resume from {resume_dir}: it was written for "
                f"{ {k: progress.get(k) for k in shape} }, this run has {shape}. Use the same dataset, "
                "filters, batch size and seed, or pass resume=False (or another output_dir)."
            )
        global_step = resume_manager.load(last, backbone, optimizer)
        if progress.get("global_step") != global_step:
            progress = {}  # killed between the checkpoint and its sidecar: the epoch's sums restart
        logger.info(
            f"Resuming at optimizer step {global_step}/{total_steps} "
            f"(epoch {global_step // n_batches_per_epoch}, batch {global_step % n_batches_per_epoch}"
            f"/{n_batches_per_epoch}) from {resume_manager.get_ckpt_path(last)}"
        )
    start_epoch, start_batch = divmod(global_step, n_batches_per_epoch)
    backbone.train()
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
    if global_step >= total_steps:
        logger.info(f"Nothing to do: {global_step} optimizer steps completed, total {total_steps}")
        _write_json(summary_path, summary)
        return summary

    def _zero_prov() -> dict:
        return {p: {"l_v": 0.0, "l_q": 0.0, "n_samples": 0} for p in PROVENANCES}

    # Optimization guards (job 753932: the Q+ loss blew up at steps 500-600 with
    # no clipping, no warmup and lr 1e-3, and stayed at twice its earlier level).
    # Group order is fixed by setup_value_optimizer: V, Q+, V_target (lr 0).
    base_lrs = [config.v_learning_rate, config.q_plus_learning_rate, 0.0]
    clip = float(config.max_grad_norm) if config.max_grad_norm else float("inf")
    slot_params = [g["params"] for g in optimizer.param_groups[:2]]

    def _set_lrs(step_now: int) -> None:
        # Re-applied every step from the config: warmup, and a learning rate changed
        # between jobs takes effect on resume (the checkpoint would restore the old one).
        scale = min(1.0, (step_now + 1) / config.warmup_steps) if config.warmup_steps else 1.0
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

    def _zero_window() -> dict:
        return {"max_l_v": 0.0, "max_l_q": 0.0, "max_gn_v": 0.0, "max_gn_q": 0.0, "clipped": 0}

    window = _zero_window()

    def _save(step_now: int, epoch_now: int, batch_in_epoch: int, sums: dict) -> None:
        # Seed V_target = V exactly so the weights-only file is usable at any checkpoint.
        backbone.polyak_update_v_target(tau=1.0)
        ckpt_manager.save(step=0, backbone=backbone, optimizer=optimizer)
        resume_manager.save(step=step_now, backbone=backbone, optimizer=optimizer)
        resume_manager.maybe_clean()
        _write_json(
            progress_path,
            {**shape, "global_step": step_now, "epoch": epoch_now, "batch_in_epoch": batch_in_epoch, **sums},
        )

    for epoch in range(start_epoch, config.max_epochs):
        skip = start_batch if epoch == start_epoch else 0
        if skip and progress:
            epoch_l_v = float(progress.get("epoch_l_v", 0.0))
            epoch_l_q = float(progress.get("epoch_l_q", 0.0))
            by_prov = _zero_prov()
            for p, d in (progress.get("by_prov") or {}).items():
                if p in by_prov:
                    by_prov[p].update(d)
            n_batches = int(progress.get("n_batches", 0))
        else:
            epoch_l_v = epoch_l_q = 0.0
            by_prov = _zero_prov()
            n_batches = 0
        for b, batch_idx in enumerate(
            _shuffled_batches(len(dataset), config.data.batch_size, config.seed + epoch)
        ):
            if b < skip:
                continue
            set_request_id(str(global_step))
            _set_lrs(global_step)
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
            # Per-slot clipping (inf = measure only): one slot's bad batch must not
            # shrink the other slot's update.
            gn_v, gn_q = (float(torch.nn.utils.clip_grad_norm_(ps, clip)) for ps in slot_params)
            window["max_l_v"] = max(window["max_l_v"], b_l_v)
            window["max_l_q"] = max(window["max_l_q"], b_l_q)
            window["max_gn_v"] = max(window["max_gn_v"], gn_v)
            window["max_gn_q"] = max(window["max_gn_q"], gn_q)
            window["clipped"] += int(gn_v > clip) + int(gn_q > clip)
            optimizer.step()
            epoch_l_v += b_l_v
            epoch_l_q += b_l_q
            n_batches += 1
            global_step += 1
            if (
                config.ckpt_every_steps
                and (b + 1) % config.ckpt_every_steps == 0
                and (b + 1) < n_batches_per_epoch
            ):
                _save(
                    global_step, epoch, b + 1,
                    {"epoch_l_v": epoch_l_v, "epoch_l_q": epoch_l_q, "by_prov": by_prov, "n_batches": n_batches},
                )
                logger.info(
                    f"checkpoint: epoch {epoch} batch {b + 1}/{n_batches_per_epoch} "
                    f"(step {global_step}/{total_steps}) running l_v={epoch_l_v / n_batches:.5f} "
                    f"l_q={epoch_l_q / n_batches:.5f} | window max batch l_v={window['max_l_v']:.4f} "
                    f"l_q={window['max_l_q']:.4f} max grad norm V={window['max_gn_v']:.3g} "
                    f"Q+={window['max_gn_q']:.3g} clipped={window['clipped']} "
                    f"lr V={optimizer.param_groups[0]['lr']:.2e} Q+={optimizer.param_groups[1]['lr']:.2e}"
                )
                window = _zero_window()
        # Epoch end: the next resume point is the start of the following epoch.
        _save(
            global_step, epoch + 1, 0,
            {"epoch_l_v": 0.0, "epoch_l_q": 0.0, "by_prov": _zero_prov(), "n_batches": 0},
        )
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
            f"-> saved value_state.pt (+ resume step {global_step}/{total_steps})"
        )
    return summary

def main() -> None:
    """Entry point: `uv run warmstart-train @ warmstart-train.toml`."""
    train(cli(WarmStartTrainConfig))


if __name__ == "__main__":
    main()
