import numpy as np
import pytest

from mlsystems.bandits import (
    bootstrap_mean_ci, epsilon_greedy, evaluate, importance_terms,
    predict_rewards, run, simulate, train_reward_model,
)


def test_synthetic_rewards_and_propensities_are_valid():
    data = simulate(n=300, seed=4)
    assert len(data) == 300
    assert np.all((data.propensities > 0) & (data.propensities <= 1))
    assert set(data.rewards).issubset({0.0, 1.0})


def test_seed_is_deterministic():
    a, b = simulate(n=300, seed=9), simulate(n=300, seed=9)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.rewards, b.rewards)


def test_target_policy_has_probability_simplex():
    scores = np.array([[.2, .8, .1], [.9, .1, .2]])
    target = epsilon_greedy(scores, epsilon=.12)
    np.testing.assert_allclose(target.sum(axis=1), 1.0)
    assert np.all(target > 0)


def test_estimates_are_finite_and_have_effective_sample_size():
    log = simulate(n=600, seed=11)
    model = train_reward_model(log.subset(slice(None, 400)))
    holdout = log.subset(slice(400, None))
    predictions = predict_rewards(holdout.contexts, model)
    target = epsilon_greedy(predictions)
    values = evaluate(holdout, target, predictions)
    assert all(np.isfinite(v) for v in values.values())
    assert 0 < values["effective_sample_size"] <= len(holdout)


def test_invalid_logged_propensity_is_rejected():
    log = simulate(n=100, seed=8)
    log.propensities[0] = 0
    policy = np.full((100, 4), .25)
    with pytest.raises(ValueError, match="propensities"):
        importance_terms(log, policy, np.full_like(policy, .5))


def test_bootstrap_is_reproducible():
    arr = np.arange(20, dtype=float)
    assert bootstrap_mean_ci(arr, seed=123) == bootstrap_mean_ci(arr, seed=123)
    low, high = bootstrap_mean_ci(arr)
    assert 0 < low < high < 20


def test_bandit_report_is_deterministic():
    assert run(seed=6, n=500) == run(seed=6, n=500)


def test_bad_epsilon_rejected():
    with pytest.raises(ValueError):
        epsilon_greedy(np.ones((2, 3)), epsilon=1.1)
