"""Logged contextual-bandit simulation, policy learning and off-policy evaluation.

The synthetic environment exposes the expected reward of each action so an
oracle comparison can be made; that information is never used to train the
reward model and would not be available with real-world logged events.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BanditLog:
    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    reward_probabilities: np.ndarray

    def __len__(self):
        return len(self.actions)

    def subset(self, subset):
        return BanditLog(self.contexts[subset], self.actions[subset],
                         self.rewards[subset], self.propensities[subset],
                         self.reward_probabilities[subset])


def _softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def simulate(n=5000, actions=4, features=6, seed=42, exploration=.25):
    if n < 100 or actions < 2 or features < 1:
        raise ValueError("n >= 100, actions >= 2 and features >= 1 required")
    if not 0 < exploration <= 1:
        raise ValueError("exploration must be in (0,1]")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features))
    weights = rng.normal(size=(features, actions)) / np.sqrt(features)
    true_values = 1 / (1 + np.exp(-(x @ weights)))
    logging = (1 - exploration) * _softmax(x @ weights * .5) + exploration / actions
    selected = np.array([rng.choice(actions, p=prob) for prob in logging], dtype=int)
    propensity = logging[np.arange(n), selected]
    rewards = rng.binomial(1, true_values[np.arange(n), selected]).astype(float)
    return BanditLog(x, selected, rewards, propensity, true_values)


def train_reward_model(log: BanditLog, actions: int = 4, regularization: float = 2.0):
    """Independent ridge reward model for each action, with an intercept."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    x = np.column_stack([np.ones(len(log)), log.contexts])
    coefs = np.zeros((x.shape[1], actions))
    for action in range(actions):
        chosen = log.actions == action
        if not np.any(chosen):
            continue
        x_action, y_action = x[chosen], log.rewards[chosen]
        penalty = np.eye(x.shape[1]) * regularization
        penalty[0, 0] = 0
        coefs[:, action] = np.linalg.solve(x_action.T @ x_action + penalty,
                                            x_action.T @ y_action)
    return coefs


def predict_rewards(contexts, coefs):
    x = np.column_stack([np.ones(len(contexts)), contexts])
    return np.clip(x @ coefs, 0.0, 1.0)


def epsilon_greedy(predictions, epsilon=.1):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be within [0,1]")
    if np.ndim(predictions) != 2 or predictions.shape[1] < 2:
        raise ValueError("predictions must be N x actions")
    n, actions = predictions.shape
    p = np.full((n, actions), epsilon / actions)
    p[np.arange(n), predictions.argmax(axis=1)] += 1 - epsilon
    return p


def _validate(log, target_probs, predicted):
    if target_probs.ndim != 2 or target_probs.shape[0] != len(log):
        raise ValueError("target policy probabilities do not match log")
    if predicted.shape != target_probs.shape:
        raise ValueError("predicted rewards and target policy shapes differ")
    if np.max(log.actions) >= target_probs.shape[1]:
        raise ValueError("logged action outside target action space")
    if np.any(log.propensities <= 0) or np.any(log.propensities > 1):
        raise ValueError("logging propensities must lie in (0,1]")
    if not np.all(np.isfinite(target_probs)) or np.any(target_probs < 0):
        raise ValueError("target policy contains invalid probabilities")
    if not np.allclose(target_probs.sum(axis=1), 1, atol=1e-8):
        raise ValueError("target action probabilities must sum to one")


def importance_terms(log: BanditLog, target_probs, predicted):
    """Return per-row OPE contributions, including doubly robust correction."""
    target_probs = np.asarray(target_probs, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    _validate(log, target_probs, predicted)
    idx = np.arange(len(log))
    weights = target_probs[idx, log.actions] / log.propensities
    ips = weights * log.rewards
    direct = (target_probs * predicted).sum(axis=1)
    dr = direct + weights * (log.rewards - predicted[idx, log.actions])
    return {"importance_weights": weights, "ips": ips, "dr": dr}


def evaluate(log: BanditLog, target_probs, predicted):
    terms = importance_terms(log, target_probs, predicted)
    weights = terms["importance_weights"]
    denominator = weights.sum()
    return {
        "ips": float(terms["ips"].mean()),
        "snips": float(terms["ips"].sum() / denominator) if denominator else float("nan"),
        "doubly_robust": float(terms["dr"].mean()),
        "effective_sample_size": float(denominator**2 / np.square(weights).sum()) if denominator else 0.0,
        "oracle": float(np.mean(np.sum(target_probs * log.reward_probabilities, axis=1))),
    }


def bootstrap_mean_ci(values, seed=0, repeats=600, confidence=.95):
    """Naive i.i.d. bootstrap; correlated events require grouped resampling."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector of length >= 2")
    if repeats < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap arguments")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(repeats)])
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(means, [alpha, 1 - alpha]))


def run(seed=42, n=5000):
    log = simulate(n=n, seed=seed)
    split = int(.6 * len(log))
    train = log.subset(slice(None, split))
    holdout = log.subset(slice(split, None))
    weights = train_reward_model(train)
    predictions = predict_rewards(holdout.contexts, weights)
    target = epsilon_greedy(predictions)
    terms = importance_terms(holdout, target, predictions)
    result = evaluate(holdout, target, predictions)
    result.update({"train_samples": len(train), "eval_samples": len(holdout),
                   "dr_95pct_bootstrap_ci": bootstrap_mean_ci(terms["dr"], seed=seed + 7),
                   "seed": seed, "synthetic_data": True})
    return result
