# Reproducibility and limitations

1. Python 3.10+; NumPy for algorithms; pytest for tests. An editable installation is sufficient.
2. The CLI uses explicit random seeds. Unit tests should pass on CPU without network access.
3. Search queries are over hand-authored toy documents; no internet corpus, embeddings API, or licensed private data.
4. Bandit outcomes are sampled from a known synthetic reward function. The oracle is **not available in real logged datasets**.
5. INT4 memory results refer to weight storage plus metadata, not total runtime peak GPU memory.
6. Agent tool traces are deterministic fixtures and do not establish an LLM model's reasoning ability or safety in production.
7. Do not copy any CLI metric into a job application without specifying this synthetic setup and performing your own verification.
