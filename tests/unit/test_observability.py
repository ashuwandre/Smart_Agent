"""Tests for request telemetry and circuit status reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from customer_ops_agent.guardrails import AgentOutput, CircuitBreaker
from customer_ops_agent.observability import (
    ObservableGraph,
    ObservabilityStore,
    circuit_state_snapshot,
)


class _SuccessfulGraph:
    def invoke(self, input, config=None, **kwargs):
        return {
            **input,
            "input_guardrail": SimpleNamespace(decision="allow"),
            "output_candidate": AgentOutput(
                response="Completed",
                action="answer",
                confidence=0.9,
            ),
            "route": "billing",
            "response": "Completed",
        }


class _FailingGraph:
    def invoke(self, input, config=None, **kwargs):
        raise RuntimeError("planner unavailable")


def test_observable_graph_logs_success_and_failure_requests(tmp_path: Path) -> None:
    store = ObservabilityStore(
        tmp_path / "tool_calls.csv",
        tmp_path / "conversation_logs.json",
    )
    successful = ObservableGraph(
        _SuccessfulGraph(),
        store,
        sanitize=lambda text: text.replace("person@example.com", "[EMAIL]"),
    )
    failing = ObservableGraph(_FailingGraph(), store, sanitize=lambda text: text)

    successful.invoke(
        {
            "conversation_id": "conversation-1",
            "customer_id": "CUST0001",
            "message": "Send it to person@example.com",
        }
    )
    with pytest.raises(RuntimeError, match="planner unavailable"):
        failing.invoke(
            {
                "conversation_id": "conversation-2",
                "message": "Second request",
            }
        )

    document = json.loads(
        (tmp_path / "conversation_logs.json").read_text(encoding="utf-8")
    )
    assert [request["status"] for request in document["requests"]] == [
        "completed",
        "error",
    ]
    assert document["requests"][0]["request"] == "Send it to [EMAIL]"
    assert document["requests"][0]["route"] == "billing"
    assert document["requests"][1]["error"] == (
        "RuntimeError: planner unavailable"
    )
    assert all(request["latency"] >= 0 for request in document["requests"])


def test_circuit_breaker_status_is_available_to_dashboard() -> None:
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=30,
        name="openai",
    )

    breaker.call(lambda: (_ for _ in ()).throw(ConnectionError("offline")))

    assert circuit_state_snapshot()["openai"] == "open"
