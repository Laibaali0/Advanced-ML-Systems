# Design decisions

**Retrieval.** BM25 is a strong lexical baseline. Truncated SVD over document TF-IDF vectors offers a small, non-neural latent representation. Reciprocal Rank Fusion combines ranked lists without pretending scores are calibrated. Held-out relevance labels must be documented before comparing ranking metrics.

**Bandits.** A known logging propensity is essential for IPS/SNIPS/DR. The simulation uses separate training and evaluation samples. Even with correct implementation, estimates may have high variance under policy mismatch. Effective sample size and bootstrap intervals are diagnostics, not online impact claims.

**Quantization.** Weights are stored in a compact, packed 4-bit representation; dequantization in NumPy happens on CPU. Memory accounting distinguishes stored weights and scale metadata from transient computation buffers. This is **not a GPU serving benchmark**.

**Agent execution.** Tools are allowlisted. Calls are validated against schema-like argument definitions and a replay budget. Side effects are disabled by default. The deterministic examples demonstrate reliability concepts rather than LLM reasoning quality.
