import pytest
from mlsystems.statistics import paired_bootstrap


def test_paired_bootstrap_matches_fixed_delta():
    before = {"one": {"ndcg": 0.9}, "two": {"ndcg": 0.8}}
    after = {"one": {"ndcg": 0.6}, "two": {"ndcg": 0.5}}
    report = paired_bootstrap(before, after, repeats=20)
    assert report["mean_difference"] == pytest.approx(.3)
    assert report["bootstrap_ci"][0] == pytest.approx(.3)


def test_pairing_requires_same_queries():
    with pytest.raises(ValueError):
        paired_bootstrap({"first": {"ndcg": 1}}, {"other": {"ndcg": 1}})


def test_missing_metric_is_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap({"first": {}}, {"first": {}}, metric="ndcg")


def test_paired_bootstrap_reproducible():
    a = {"one": {"ndcg": .4}, "two": {"ndcg": .9}}
    b = {"one": {"ndcg": .6}, "two": {"ndcg": .6}}
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)
