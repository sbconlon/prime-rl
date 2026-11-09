"""Warm-start dataset (Phase 3): reads the Phase-2 records and applies
build_warmstart_samples to produce a flat list of AdvantageTrainingSamples
(targets = MC returns) for the offline value regression."""

from __future__ import annotations

from pathlib import Path

from prime_rl.transport.types import AdvantageTrainingSample
from prime_rl.value_warmstart.dataset_io import read_branch_records, read_records
from prime_rl.value_warmstart.targets import build_warmstart_samples


class WarmStartDataset:
    """In-memory list of warm-start AdvantageTrainingSamples.

    Each WarmStartRecord is expanded via build_warmstart_samples (is_terminal =
    not is_truncated) and flattened. If the dataset does not fit in memory, swap
    this for a streaming variant with the same __len__ / __getitem__ contract.
    """

    def __init__(self, path: Path, *, gamma: float = 1.0, n_step: int | None = None):
        self._samples: list[AdvantageTrainingSample] = []
        for record in read_records(path):
            self._samples.extend(
                build_warmstart_samples(
                    record.samples,
                    episodic_reward=record.reward,
                    is_terminal=not record.is_truncated,
                    gamma=gamma,
                    n_step=n_step,
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AdvantageTrainingSample:
        return self._samples[idx]


class BranchedDataset:
    """In-memory list of AdvantageTrainingSamples from a branched (Job-B) dataset.

    Each BranchRecord is expanded via build_warmstart_samples with the record's
    label_from_dp / root_dp (plan branched-dataset D-10): a branch's pre-root
    prefix gets no targets, the forced root labels Q+ only (v_weight = 0). A
    legacy WarmStartRecord file loads too (trunk-like records, read_branch_records).

    Held-out games (record.holdout, or game_id in `holdout_ids`) are excluded
    unless include_holdout; their ids are kept in `holdout_game_ids` for
    holdout.json. Samples whose target list comes out empty (all decision points
    precede the root) are dropped: the legs would forward them for nothing, and no
    other sample depends on them (each row is forwarded independently).

    `provenance[i]` is "trunk" or "branch" for sample i (per-provenance loss
    logging in warmstart-train); `n_root_targets` counts forced roots kept.
    """

    def __init__(
        self,
        path: Path,
        *,
        gamma: float = 1.0,
        n_step: int | None = None,
        holdout_ids: set[str] | None = None,
        include_holdout: bool = False,
    ):
        holdout_ids = set(holdout_ids or ())
        self._samples: list[AdvantageTrainingSample] = []
        self.provenance: list[str] = []
        self.holdout_game_ids: set[str] = set()
        self.n_records = 0
        self.n_holdout_records = 0
        self.n_root_targets = 0
        self.n_dropped_empty = 0
        for record in read_branch_records(path):
            self.n_records += 1
            is_holdout = bool(record.holdout) or record.game_id in holdout_ids
            if is_holdout:
                self.n_holdout_records += 1
                self.holdout_game_ids.add(record.game_id)
                if not include_holdout:
                    continue
            samples = build_warmstart_samples(
                record.samples,
                episodic_reward=record.reward,
                is_terminal=not record.is_truncated,
                gamma=gamma,
                n_step=n_step,
                label_from_dp=record.label_from_dp,
                root_dp=record.root_dp,
            )
            prov = "branch" if record.root_forced else "trunk"
            if record.root_forced and record.root_dp is not None:
                self.n_root_targets += 1
            for s in samples:
                if not s.decision_point_targets:
                    self.n_dropped_empty += 1
                    continue
                self._samples.append(s)
                self.provenance.append(prov)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AdvantageTrainingSample:
        return self._samples[idx]
