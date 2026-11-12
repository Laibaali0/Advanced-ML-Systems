"""Deterministic, allowlisted agent tool-call replay and reliability evaluation.

This module does not call an LLM. It evaluates example traces against tool
contracts, execution budgets, expected outputs, and write-operation gates.
"""
from dataclasses import dataclass
from typing import Callable
import json


class ToolError(Exception):
    """Error surfaced to a tool-calling agent without exposing internals."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arguments: dict[str, type]
    handler: Callable
    side_effect: bool = False


class ToolRuntime:
    def __init__(self, tools: list[ToolSpec], max_calls: int = 6, allow_writes: bool = False):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("duplicate tool names")
        self.tools = {tool.name: tool for tool in tools}
        self.max_calls = max_calls
        self.allow_writes = allow_writes
        self.audit: list[dict] = []
        self.calls = 0

    def call(self, name: str, arguments: dict):
        """Validate every call; log failures as well as successful executions."""
        self.calls += 1
        record = {"tool": name, "arguments": arguments, "status": "blocked"}
        try:
            if self.calls > self.max_calls:
                raise ToolError("Tool call budget exceeded")
            if name not in self.tools:
                raise ToolError("Unknown tool")
            spec = self.tools[name]
            if spec.side_effect and not self.allow_writes:
                raise ToolError("Write tool not permitted in read-only mode")
            if not isinstance(arguments, dict) or set(arguments) != set(spec.arguments):
                raise ToolError("Tool arguments do not match the required schema")
            for key, required_type in spec.arguments.items():
                value = arguments[key]
                # bool is a subclass of int in Python, so disallow it explicitly.
                if isinstance(value, bool) and required_type is int:
                    raise ToolError(f"Argument {key} must be {required_type.__name__}")
                if not isinstance(value, required_type):
                    raise ToolError(f"Argument {key} must be {required_type.__name__}")
            result = spec.handler(**arguments)
            json.dumps(result)  # ensure traces remain serializable
            record.update({"status": "ok", "result": result})
            return result
        except ToolError as error:
            record["error"] = str(error)
            raise
        except Exception as error:
            record["error"] = "Tool execution failed"
            raise ToolError("Tool execution failed") from error
        finally:
            self.audit.append(record)


def default_runtime(max_calls=6):
    catalog = {
        "SKU-1": {"title": "Running Shoes", "price_cents": 7999},
        "SKU-2": {"title": "Canvas Bag", "price_cents": 2499},
    }

    def catalog_lookup(sku):
        if sku not in catalog:
            raise ToolError("Item not found")
        return catalog[sku]

    def add_numbers(a, b):
        return a + b

    def delete_item(sku):
        # This demonstration should never have write permission in the evaluator.
        return {"deleted": sku}

    return ToolRuntime([
        ToolSpec("catalog_lookup", {"sku": str}, catalog_lookup),
        ToolSpec("add_numbers", {"a": int, "b": int}, add_numbers),
        ToolSpec("delete_item", {"sku": str}, delete_item, side_effect=True),
    ], max_calls=max_calls, allow_writes=False)


def evaluate_cases(cases, runtime_factory=default_runtime):
    """Evaluate predetermined traces and expected outputs, not model quality.

    Each case has a list of calls: {'tool': ..., 'arguments': ..., 'expected': ...}.
    An expected_error substring is allowed instead of expected for blocked calls.
    """
    reports = []
    for case in cases:
        runtime = runtime_factory()
        successes = 0
        for call in case["calls"]:
            try:
                output = runtime.call(call["tool"], call["arguments"])
                passed = "expected_error" not in call and output == call.get("expected")
            except ToolError as error:
                passed = bool(call.get("expected_error")) and call["expected_error"] in str(error)
            successes += bool(passed)
        reports.append({
            "case": case["name"],
            "passed": successes == len(case["calls"]),
            "passed_calls": successes,
            "total_calls": len(case["calls"]),
            "blocked_calls": sum(event["status"] == "blocked" for event in runtime.audit),
        })
    return {
        "cases": len(reports),
        "passed_cases": sum(report["passed"] for report in reports),
        "pass_rate": sum(report["passed"] for report in reports) / len(reports) if reports else 0.0,
        "results": reports,
        "is_deterministic_trace_evaluation": True,
    }


DEFAULT_CASES = [
    {"name": "catalog read", "calls": [
        {"tool": "catalog_lookup", "arguments": {"sku": "SKU-1"},
         "expected": {"title": "Running Shoes", "price_cents": 7999}}]},
    {"name": "arithmetic", "calls": [
        {"tool": "add_numbers", "arguments": {"a": 7, "b": 5}, "expected": 12}]},
    {"name": "reject unauthorized mutation", "calls": [
        {"tool": "delete_item", "arguments": {"sku": "SKU-1"},
         "expected_error": "not permitted"}]},
    {"name": "reject malformed arguments", "calls": [
        {"tool": "add_numbers", "arguments": {"a": "7", "b": 5},
         "expected_error": "must be int"}]},
]
