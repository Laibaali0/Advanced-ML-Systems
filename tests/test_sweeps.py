import pytest
from mlsystems.sweeps import bandit_seed_sweep, group_size_sweep


def test_bandit_sweep_is_reproducible():
    first = bandit_seed_sweep([2, 3], samples=300)
    second = bandit_seed_sweep([2, 3], samples=300)
    assert first == second
    assert set(first["estimators"]) == {"ips", "snips", "doubly_robust"}


def test_group_sweep_preserves_shape_and_shows_storage_tradeoff():
    result = group_size_sweep(shape=(32, 33), groups=(4, 8, 16))
    sizes = [entry["weight_storage_bytes"] for entry in result["results"]]
    assert all(s > 0 for s in sizes)
    assert sizes[0] > sizes[1] > sizes[2]


def test_empty_sweep_raises():
    with pytest.raises(ValueError):
        bandit_seed_sweep([])
    with pytest.raises(ValueError):
        group_size_sweep(groups=[])
