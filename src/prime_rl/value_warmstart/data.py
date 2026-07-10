"""Warm-start dataset (Phase 3): reads the Phase-2 records and applies
build_warmstart_samples to produce a flat list of AdvantageTrainingSamples
(targets = MC returns) for the offline value regression."""

from __future__ import annotations

from pathlib import Path

from prime_rl.transport.types import AdvantageTrainingSample
from prime_rl.value_warmstart.dataset_io import read_records
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
