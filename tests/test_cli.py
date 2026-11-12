import json
from mlsystems.cli import main, run_retrieval


def test_cli_writes_json_report(tmp_path, capsys):
    location = tmp_path / "results" / "agent.json"
    main(["agent", "--output", str(location)])
    data = json.loads(location.read_text())
    assert data["report"]["passed_cases"] == 4
    assert "agent" in capsys.readouterr().out


def test_retrieval_cli_includes_all_baselines():
    results = run_retrieval()
    assert set(results["metrics"]) == {"bm25", "latent", "hybrid"}


def test_cli_rejects_unknown_experiment():
    import pytest
    with pytest.raises(SystemExit):
        main(["not-an-experiment"])


def test_trace_option_is_agent_only():
    import pytest
    with pytest.raises(SystemExit):
        main(["retrieval", "--traces", "examples/agent_traces.json"])
