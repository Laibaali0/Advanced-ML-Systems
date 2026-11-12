import numpy as np
import pytest

from mlsystems.fixtures import retrieval_fixture
from mlsystems.ranking_metrics import ndcg_at_k, recall_at_k, reciprocal_rank, evaluate_queries
from mlsystems.retrieval import BM25, Document, HybridRetriever, LatentSearch, order, reciprocal_rank_fusion


def test_token_ranking_prefers_exact_terms():
    corpus = [Document("a", "red hiking shoes"), Document("b", "blue winter coat")]
    ranking = order(BM25(corpus).scores("hiking shoes"), corpus)
    assert ranking[0] == "a"


def test_hybrid_search_returns_unique_ids():
    corpus, _ = retrieval_fixture()
    searcher = HybridRetriever(corpus)
    for method in ("bm25", "latent", "hybrid"):
        ids = searcher.search("GPU inference", method, k=6)
        assert len(ids) == len(set(ids)) == 6
        assert set(ids).issubset({doc.id for doc in corpus})


def test_svd_vectors_and_unknown_query_are_finite():
    corpus, _ = retrieval_fixture()
    search = LatentSearch(corpus, dimensions=5)
    assert np.all(np.isfinite(search.scores("quantized transformer")))
    assert np.allclose(search.scores("never-seen-nonsense"), 0)


def test_rank_fusion_rewards_agreement():
    ranked = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    assert ranked == ["a", "b", "c"]  # deterministic alphabetical tie-break


def test_ranking_metrics_known_answers():
    judgments = {"a": 3, "b": 1, "c": 0}
    assert reciprocal_rank(["c", "b", "a"], judgments) == .5
    assert recall_at_k(["c", "b", "a"], judgments, 2) == .5
    assert ndcg_at_k(["a", "b"], judgments, 2) == pytest.approx(1)
    assert 0 < ndcg_at_k(["c", "b"], judgments, 2) < 1


def test_corpus_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        BM25([Document("x", "first"), Document("x", "second")])


def test_invalid_search_options_rejected():
    corpus, _ = retrieval_fixture()
    with pytest.raises(ValueError):
        HybridRetriever(corpus).search("hello", method="magic")
    with pytest.raises(ValueError):
        HybridRetriever(corpus).search("hello", k=0)


def test_metric_report_has_all_queries():
    corpus, judgments = retrieval_fixture()
    results = evaluate_queries(HybridRetriever(corpus), judgments, "hybrid", k=5)
    assert len(results["per_query"]) == len(judgments)
    assert all(0 <= value <= 1 for value in results["mean"].values())
