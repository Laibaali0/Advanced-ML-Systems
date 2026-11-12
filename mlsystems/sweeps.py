"""Deterministic parameter sweeps for the educational ML reference modules."""
from statistics import mean, pstdev
import numpy as np

from .bandits import run as run_bandit
from .quantization import quantize, dequantize


def bandit_seed_sweep(seeds=(11, 12, 13, 14, 15), samples=2000):
    """Compare absolute estimation error against the synthetic oracle."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    results = [run_bandit(seed=int(seed), n=samples) for seed in seeds]
    estimators = ("ips", "snips", "doubly_robust")
    summary = {}
    for estimator in estimators:
        errors = [value[estimator] - value["oracle"] for value in results]
        absolute = [abs(error) for error in errors]
        summary[estimator] = {
            "mean_absolute_oracle_error": mean(absolute),
            "mean_signed_oracle_error": mean(errors),
            "std_signed_error": pstdev(errors),
        }
    return {"seeds": seeds, "samples_per_seed": samples,
            "synthetic_oracle_available": True, "estimators": summary}


def group_size_sweep(seed=42, shape=(128, 129), groups=(8, 16, 32, 64)):
    """Compare storage/error as INT4 group size varies on the same weights."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must contain two positive dimensions")
    if not groups:
        raise ValueError("At least one group size is required")
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=shape).astype(np.float32)
    results = []
    for group in groups:
        packed = quantize(weights, group_size=group)
        reconstructed = dequantize(packed)
        results.append({
            "group_size": group,
            "weight_storage_bytes": packed.stored_bytes,
            "compression_ratio_vs_fp32": packed.fp32_weight_bytes / packed.stored_bytes,
            "mean_absolute_weight_error": float(np.mean(np.abs(weights - reconstructed))),
        })
    return {"seed": seed, "shape": list(shape), "synthetic_data": True, "results": results}
