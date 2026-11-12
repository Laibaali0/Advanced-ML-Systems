# Evaluation checklist

This project separates **correctness tests**, **synthetic offline metrics**, and **external performance claims**. Each has a different level of evidence.

## Hybrid retrieval

- Validate indexing and ranking deterministically with small fixtures.
- Report BM25, low-rank SVD and RRF separately, not a selective single best run.
- Store the full relevance judgments and train/test split if using external corpora.
- Paired bootstrap samples **queries**, preserving within-query method comparisons; 5 toy queries are too few for strong conclusions.
- Don't label these SVD vectors as pretrained transformer embeddings.

## Logged bandits

- Train reward model strictly on a training split, evaluate on a disjoint split.
- Validate logging action probabilities are non-zero and target policy distributions sum to one.
- Report IPS, SNIPS, DR, and **effective sample size**. These can disagree on small samples.
- For repeat interactions, cluster resampling by user/session—not per interaction—as a next improvement.
- Synthetic oracle value is shown for diagnostics only; it does not exist for actual observational commerce data.

## Groupwise INT4

- Check odd-width matrices, zero rows, and non-finite input rejection.
- Compare storage bytes including scale metadata rather than misleading 4:1 ratios for every tensor.
- Report numerical matrix and output errors under fixed input seeds.
- CPU timings are **not** CUDA kernel or production LLM serving benchmarks.

## Agent traces

- Evaluate tool argument schemas and explicit write denials.
- Reject unknown tools and enforce a request budget.
- Measure trace-replay outcomes separately from model-generated plans.
- Never use tool side effects or private APIs in this reference suite.

## Follow-on benchmark work

A real research-grade report would need a specified public dataset, hardware, baseline, model/checkpoint versions, provenance, confidence intervals, and enough samples to support a meaningful claim. Track failures and negative comparisons as carefully as wins.
