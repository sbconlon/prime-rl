"""The V/Q+ regression legs (value_regression_backward) and the per-slot optimizer
(setup_value_optimizer) are exercised through the existing
tests/unit/advantage_trainer suite, which now routes through
prime_rl.advantage_trainer.shared. Here we cover the offline loop's own logic."""

from prime_rl.value_warmstart.train import _shuffled_batches


def test_shuffled_batches_covers_all_indices():
    batches = list(_shuffled_batches(10, 3, seed=0))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(10))
    assert all(1 <= len(b) <= 3 for b in batches)


def test_shuffled_batches_deterministic_per_seed():
    assert list(_shuffled_batches(10, 3, seed=0)) == list(_shuffled_batches(10, 3, seed=0))


def test_shuffled_batches_varies_by_seed():
    assert list(_shuffled_batches(20, 3, seed=0)) != list(_shuffled_batches(20, 3, seed=1))
