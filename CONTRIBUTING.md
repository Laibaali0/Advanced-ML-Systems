# Contributing and extending the experiments

These reference implementations prioritize correctness, reproducibility, and honest documentation over inflated benchmark claims.

1. Add or edit an algorithm under `mlsystems/` and keep its interface small.
2. Include a deterministic, focused regression test. Cover invalid inputs and a synthetic fixture with an expected outcome.
3. Run `python -m pytest -q` and relevant CLI commands.
4. If comparing methods, use the same test set, seed, and hardware. Include baseline numbers and negative results.
5. Keep user- or employer-owned data, credentials, API tokens, and proprietary code out of the repository.
6. Describe what was actually implemented; mark future ideas as planned, not shipped.

This project was developed with AI assistance. Review, test, and understand changes before presenting them as your work.
