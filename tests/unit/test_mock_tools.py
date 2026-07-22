"""Behavior and observability tests for the mock tool boundary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from customer_ops_agent.observability import ObservabilityStore
from customer_ops_agent.tools import audit, mock_tools


@pytest.fixture()
def tool_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate mutable tool state and audit output for every test."""

    mock_tools._reset_mock_state()
    log_path = tmp_path / "mock_tools.jsonl"
    monkeypatch.setattr(audit, "TOOL_LOG_PATH", log_path)
    monkeypatch.setattr(
        audit,
        "TOOL_OBSERVABILITY_STORE",
        ObservabilityStore(
            tmp_path / "tool_calls.csv",
            tmp_path / "conversation_logs.json",
        ),
    )
    return log_path


def test_customer_retention_refund_ticket_and_escalation_tools(
    tool_log: Path,
) -> None:
    """Core tools return typed outcomes while enforcing financial controls."""

    customer = mock_tools.get_customer("cust0058")
    offer = mock_tools.grant_retention_offer("CUST0058", 10)
    refund = mock_tools.issue_refund("CUST0094", 10_000)
    ticket = mock_tools.create_ticket("CUST0102", "Buffering after Wi-Fi checks")
    escalation = mock_tools.escalate_to_human(
        "Refund requires approval",
        "CUST0094",
    )

    assert customer.success and customer.customer is not None
    assert customer.customer.customer_id == "CUST0058"
    assert offer.status == "granted"
    assert offer.offer_cost == 179.7
    assert refund.status == "pending_approval"
    assert refund.transaction_id is None
    assert ticket.ticket_id == "TKT000001"
    assert escalation.escalation_id == "ESC000001"

    events = [
        json.loads(line)
        for line in tool_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["tool"] for event in events] == [
        "get_customer",
        "grant_retention_offer",
        "issue_refund",
        "create_ticket",
        "escalate_to_human",
    ]
    assert all("input" in event and "output" in event for event in events)
    assert all(event["execution_time_ms"] >= 0 for event in events)
    assert all("error" in event for event in events)

    with (tool_log.parent / "tool_calls.csv").open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert reader.fieldnames == [
            "timestamp",
            "customer",
            "tool",
            "confidence",
            "latency",
            "retry",
            "status",
            "loop_count",
            "reason",
        ]
    assert len(rows) == 5


def test_search_kb_returns_three_structured_chunks(
    tool_log: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool delegates retrieval and preserves citation metadata."""

    monkeypatch.setattr(mock_tools, "rag_build_index", lambda: 10)
    monkeypatch.setattr(
        mock_tools,
        "rag_search",
        lambda query: [
            {
                "content": f"Grounded result {position} for {query}",
                "score": 0.9 - position / 10,
                "metadata": {
                    "filename": "refund_policy.md",
                    "chunk_id": f"refund_policy:{position:04d}",
                    "title": "Refund Policy",
                },
            }
            for position in range(1, 4)
        ],
    )

    result = mock_tools.search_kb("refund approval threshold")

    assert result.success
    assert len(result.chunks) == 3
    assert result.chunks[0].metadata.filename == "refund_policy.md"

    event = json.loads(tool_log.read_text(encoding="utf-8").splitlines()[0])
    assert event["tool"] == "search_kb"
    assert event["input"] == {"query": "refund approval threshold"}
    assert len(event["output"]["chunks"]) == 3


def test_expected_tool_error_is_structured_and_logged(tool_log: Path) -> None:
    """Missing records produce a typed failure and an observable error field."""

    result = mock_tools.get_customer("CUST9999")

    assert not result.success
    assert result.error == "Customer CUST9999 was not found"

    event = json.loads(tool_log.read_text(encoding="utf-8").splitlines()[0])
    assert event["error"] == result.error
