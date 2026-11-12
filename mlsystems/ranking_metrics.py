"""Transparent query-level retrieval metrics with graded relevance support."""
from math import log2


def reciprocal_rank(ranked: list[str], judgments: dict[str, int]) -> float:
    for rank, identifier in enumerate(ranked, start=1):
        if judgments.get(identifier, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    positive = {identifier for identifier, value in judgments.items() if value > 0}
    if not positive:
        return 0.0
    return len(positive.intersection(ranked[:k])) / len(positive)


def ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if any(value < 0 for value in judgments.values()):
        raise ValueError("relevance judgments must be non-negative")
    dcg = sum((2**judgments.get(identifier, 0) - 1) / log2(rank + 1)
              for rank, identifier in enumerate(ranked[:k], start=1))
    ideal_values = sorted(judgments.values(), reverse=True)[:k]
    ideal = sum((2**value - 1) / log2(rank + 1)
                for rank, value in enumerate(ideal_values, start=1))
    return dcg / ideal if ideal else 0.0


def evaluate_queries(retriever, queries: dict[str, dict[str, int]], method: str, k: int = 5):
    if not queries:
        raise ValueError("provide at least one labeled query")
    per_query = {}
    for query, judgments in queries.items():
        ranked = retriever.search(query, method=method, k=k)
        per_query[query] = {
            "recall": recall_at_k(ranked, judgments, k),
            "mrr": reciprocal_rank(ranked, judgments),
            "ndcg": ndcg_at_k(ranked, judgments, k),
        }
    means = {metric: sum(row[metric] for row in per_query.values()) / len(per_query)
             for metric in ("recall", "mrr", "ndcg")}
    return {"k": k, "method": method, "mean": means, "per_query": per_query}
