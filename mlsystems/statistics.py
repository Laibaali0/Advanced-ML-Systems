"""Bootstrap difference intervals for *paired* query-level evaluation scores.

For small, hand-authored query sets these intervals are only illustrative.
Do not treat them as evidence of a statistically significant production lift.
"""
import numpy as np


def paired_bootstrap(first, second, metric="ndcg", seed=42, repeats=1000, confidence=.95):
    """Resample queries (not individual results), preserving matched pairs.

    first and second: results from evaluate_queries(...)["per_query"].
    """
    if not first or set(first) != set(second):
        raise ValueError("Both methods need exactly the same labeled queries")
    if repeats < 2 or not 0 < confidence < 1:
        raise ValueError("repeats >= 2 and 0 < confidence < 1 required")
    keys = sorted(first)
    try:
        delta = np.array([float(first[key][metric]) - float(second[key][metric]) for key in keys])
    except KeyError as error:
        raise ValueError("Requested metric is missing") from error
    if not np.isfinite(delta).all():
        raise ValueError("Metric values must be finite")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(repeats, len(keys)))
    sampled = delta[draws].mean(axis=1)
    low, high = np.quantile(sampled, [(1-confidence)/2, 1-(1-confidence)/2])
    return {"metric": metric, "queries": len(keys),
            "mean_difference": float(delta.mean()),
            "bootstrap_ci": [float(low), float(high)],
            "seed": seed, "resamples": repeats}
