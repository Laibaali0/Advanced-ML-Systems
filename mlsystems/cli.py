"""Small command-line interface for reproducible demonstrations."""
import argparse
import json
from pathlib import Path

from .agent_eval import DEFAULT_CASES, evaluate_cases
from .bandits import run as run_bandit
from .fixtures import retrieval_fixture
from .quantization import compare
from .ranking_metrics import evaluate_queries
from .retrieval import HybridRetriever
from .statistics import paired_bootstrap
from .sweeps import bandit_seed_sweep, group_size_sweep


def run_retrieval(k=5, dimensions=6):
    docs, judgments = retrieval_fixture()
    retriever = HybridRetriever(docs, dimensions=dimensions)
    reports = {method: evaluate_queries(retriever, judgments, method, k=k)
               for method in ("bm25", "latent", "hybrid")}
    comparison = paired_bootstrap(reports["hybrid"]["per_query"],
                                  reports["bm25"]["per_query"], metric="ndcg")
    return {"dataset": "hand-authored toy corpus", "documents": len(docs),
            "queries": len(judgments),
            "metrics": {method: report["mean"] for method, report in reports.items()},
            "hybrid_minus_bm25_paired_bootstrap": comparison}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("retrieval", "bandit", "quantization", "agent", "sweeps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, help="write JSON report to a local file")
    parser.add_argument("--traces", type=Path, help="agent-only JSON trace replay fixture")
    args = parser.parse_args(argv)
    if args.traces and args.experiment != "agent":
        parser.error("--traces is only supported for the agent experiment")

    if args.experiment == "retrieval":
        report = run_retrieval()
    elif args.experiment == "bandit":
        report = run_bandit(seed=args.seed, n=args.samples)
    elif args.experiment == "quantization":
        report = compare(seed=args.seed)
    elif args.experiment == "agent":
        if args.traces:
            loaded = json.loads(args.traces.read_text(encoding="utf-8"))
            if not isinstance(loaded, list) or not all(isinstance(case, dict) for case in loaded):
                parser.error("--traces must contain a JSON list of case objects")
            cases = loaded
        else:
            cases = DEFAULT_CASES
        report = evaluate_cases(cases)
    else:
        report = {"bandit": bandit_seed_sweep(samples=max(200, args.samples // 2)),
                  "groupwise_int4": group_size_sweep(seed=args.seed)}

    payload = {"experiment": args.experiment, "report": report}
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return payload


if __name__ == "__main__":
    main()
