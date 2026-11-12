# Advanced ML Systems Lab

A **hands-on portfolio of reference implementations** for retrieval, offline policy evaluation, inference quantization, and reliable agent execution. These programs are designed to be understood, tested, and extended—not to impersonate production services or published research.

**Reality check:** All bundled data is synthetic or hand-authored. There are no pretrained commercial models, proprietary datasets, GPU inference kernels, enterprise deployments, or validated real-world performance claims.

## Tracks

| Area | Implementation | What to try |
| --- | --- | --- |
| Hybrid retrieval | BM25 + truncated SVD latent vectors + Reciprocal Rank Fusion | Compare lexical, latent, and hybrid search using nDCG/MRR/Recall |
| Offline bandits | Logged propensities, epsilon-greedy decisions, IPS/SNIPS/DR, bootstrapped intervals | Compare estimated policy value with a known synthetic oracle |
| 4-bit inference | Groupwise signed INT4 quantization, packed nibbles, numerical and memory error | Compare a quantized linear layer with its FP32 baseline |
| Tool-using agents | JSON-schema-like contracts, allowlisted tools, replay traces, read-only safety gates | Measure deterministic task completion and tool-call failures |

## Run locally

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m mlsystems.cli retrieval
python -m mlsystems.cli bandit
python -m mlsystems.cli quantization
python -m mlsystems.cli agent
python -m mlsystems.cli agent --traces examples/agent_traces.json
python -m mlsystems.cli sweeps
```

Each command prints results calculated **during execution** and a seed where relevant. Results should not be quoted as external benchmarks.

## Repository structure

- `mlsystems/` — tested, reusable core algorithms
- `examples/` — tiny, documented demonstration inputs
- `tests/` — correctness, regression, and edge-case checks
- `docs/` — design decisions, limits, and roadmap
- `.github/workflows/` — CI verification

See [design notes](docs/design.md), [evaluation checklist](docs/evaluation.md), and [reproducibility notes](docs/reproducibility.md) for algorithms and evidence limits. The agent evaluator accepts [replay traces](examples/agent_traces.json) you can inspect and customize.

## Ethics and attribution

This code is an original **educational implementation assembled with AI assistance**. Being present in a GitHub portfolio does not mean that it was independently authored from scratch, used by an employer, or measured on company data. Avoid adding claims to a résumé that are not supported by reproducible runs and your own understanding.
