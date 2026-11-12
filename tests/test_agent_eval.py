import pytest

from mlsystems.agent_eval import (
    DEFAULT_CASES, ToolError, ToolRuntime, ToolSpec, default_runtime, evaluate_cases,
)


def test_read_only_fixture_passes():
    report = evaluate_cases(DEFAULT_CASES)
    assert report["cases"] == 4
    assert report["pass_rate"] == 1.0


def test_unknown_tool_is_blocked_and_audited():
    runtime = default_runtime()
    with pytest.raises(ToolError, match="Unknown"):
        runtime.call("upload_secrets", {})
    assert runtime.audit[-1]["status"] == "blocked"


def test_schema_rejects_wrong_type():
    runtime = default_runtime()
    with pytest.raises(ToolError, match="must be int"):
        runtime.call("add_numbers", {"a": "1", "b": 2})


def test_bool_does_not_pass_as_int():
    runtime = default_runtime()
    with pytest.raises(ToolError, match="must be int"):
        runtime.call("add_numbers", {"a": True, "b": 2})


def test_write_denied_even_if_schema_is_valid():
    runtime = default_runtime()
    with pytest.raises(ToolError, match="not permitted"):
        runtime.call("delete_item", {"sku": "SKU-1"})


def test_budget_counts_failed_calls():
    runtime = default_runtime(max_calls=1)
    assert runtime.call("add_numbers", {"a": 1, "b": 2}) == 3
    with pytest.raises(ToolError, match="budget"):
        runtime.call("add_numbers", {"a": 1, "b": 2})


def test_duplicate_tool_names_disallowed():
    tool = ToolSpec("noop", {}, lambda: None)
    with pytest.raises(ValueError):
        ToolRuntime([tool, tool])


def test_replay_detects_wrong_expected_output():
    cases = [{"name": "incorrect", "calls": [
        {"tool": "add_numbers", "arguments": {"a": 1, "b": 2}, "expected": 4}
    ]}]
    report = evaluate_cases(cases)
    assert report["passed_cases"] == 0


def test_repository_example_traces(capsys):
    from pathlib import Path
    from mlsystems.cli import main
    example = Path(__file__).parent.parent / "examples" / "agent_traces.json"
    result = main(["agent", "--traces", str(example)])
    assert result["report"]["passed_cases"] == 3
    assert result["report"]["pass_rate"] == 1.0
