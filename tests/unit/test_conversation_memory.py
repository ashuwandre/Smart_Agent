"""Tests for bounded JSON memory and multi-turn graph context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_ops_agent.agent import (
    MockSpecialistHandler,
    RouteDecision,
    build_graph,
)
from customer_ops_agent.memory import ConversationMemoryStore
from customer_ops_agent.observability import ObservabilityStore


def test_json_store_keeps_customer_state_and_last_ten_interactions(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "conversation_memory.json"
    store = ConversationMemoryStore(memory_path)

    for position in range(12):
        store.append_interaction(
            "conversation-1",
            "customer" if position % 2 == 0 else "assistant",
            f"message-{position}",
            customer_id="CUST0058",
        )
    store.record_offer("conversation-1", 10, "granted")
    store.record_refund("conversation-1", 299, "requested")
    store.record_technical_step("conversation-1", "Restarted router")

    # A new instance proves memory is loaded from JSON rather than process state.
    reloaded = ConversationMemoryStore(memory_path).get_memory("conversation-1")

    assert reloaded.customer_id == "CUST0058"
    assert len(reloaded.interactions) == 10
    assert reloaded.interactions[0].message == "message-2"
    assert reloaded.offer_given is not None
    assert reloaded.offer_given.percentage == 10
    assert reloaded.refund_requested is not None
    assert reloaded.refund_requested.amount == 299
    assert reloaded.technical_steps == ["Restarted router"]

    raw_document = json.loads(memory_path.read_text(encoding="utf-8"))
    assert raw_document["version"] == 1
    assert "conversation-1" in raw_document["conversations"]


def test_conversation_cannot_switch_customer_identity(tmp_path: Path) -> None:
    store = ConversationMemoryStore(tmp_path / "memory.json")
    store.append_interaction(
        "conversation-1",
        "customer",
        "First request",
        customer_id="CUST0001",
    )

    with pytest.raises(ValueError, match="already belongs"):
        store.append_interaction(
            "conversation-1",
            "customer",
            "Different customer",
            customer_id="CUST0002",
        )


def test_graph_supplies_previous_turns_and_restores_customer_id(
    tmp_path: Path,
) -> None:
    planner_inputs: list[str] = []

    def planner(message: str) -> RouteDecision:
        planner_inputs.append(message)
        return RouteDecision(route="billing", reason="Billing conversation")

    store = ConversationMemoryStore(tmp_path / "memory.json")
    graph = build_graph(
        planner=planner,
        specialist_handler=MockSpecialistHandler(),
        memory_store=store,
        observability_store=ObservabilityStore(
            tmp_path / "tool_calls.csv",
            tmp_path / "conversation_logs.json",
        ),
    )

    graph.invoke(
        {
            "conversation_id": "conversation-1",
            "customer_id": "CUST0037",
            "message": "Why is this month's bill higher?",
        }
    )
    second_result = graph.invoke(
        {
            "conversation_id": "conversation-1",
            "message": "What was my previous question?",
        }
    )

    assert len(planner_inputs) == 2
    assert "Previous conversation:" in planner_inputs[1]
    assert "Why is this month's bill higher?" in planner_inputs[1]
    assert "Current customer message:\nWhat was my previous question?" in planner_inputs[1]
    assert second_result["customer_id"] == "CUST0037"

    persisted = store.get_memory("conversation-1")
    assert len(persisted.interactions) == 4
