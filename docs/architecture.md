# System architecture

The modules intentionally separate data simulation, model/policy logic, evaluation, and user-facing reports.

```mermaid
flowchart LR
  CLI[CLI and seed/config] --> R[Hybrid search]
  CLI --> B[Logged bandit simulation]
  CLI --> Q[INT4 weight packing]
  CLI --> A[Tool trace replay]
  R --> M[Ranking metrics + query bootstrap]
  B --> O[IPS / SNIPS / DR + bootstrap]
  Q --> N[Storage + numerical error]
  A --> G[Schema + permission gates]
  M --> JSON[JSON report]
  O --> JSON
  N --> JSON
  G --> JSON
```

## Module boundaries

- `retrieval.py` owns lexical and low-rank indexing and RRF, and is reusable independently of the demo corpus.
- `ranking_metrics.py` is agnostic to the ranking model, allowing direct baseline comparisons.
- `statistics.py` resamples paired query-level differences, not raw documents.
- `bandits.py` maintains distinct logging-policy, reward-model, target-policy, and evaluation steps.
- `quantization.py` keeps storage representation distinct from transient CPU dequantization.
- `agent_eval.py` validates declared tools and produces an auditable, side-effect-free replay summary.
- `sweeps.py` holds evaluation scenarios so experiments use fixed inputs for fair comparisons.

The intentionally small codebase is suitable for experimenting on a laptop. Scaling to real ML systems would require dataset lifecycle controls, streaming I/O, accelerated compute, serving endpoints, deployment observability, and security review.
