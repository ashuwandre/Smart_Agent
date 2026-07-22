"""Tests proving specialists execute model-selected tools through controls."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

from customer_ops_agent.agent.handlers import LLMToolCallingHandler
from customer_ops_agent.business_rules.models import (
    ActionExecutionResult,
    ActionValidationResult,
    RuleReason,
)
from customer_ops_agent.guardrails import evaluate_input
from customer_ops_agent.memory import MemoryContext


def _tool_message(call_id: str, name: str, arguments: dict[str, Any]):
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments),
                ),
            )
        ],
    )


def _final_message(
    response: str,
    *,
    confidence: float = 0.9,
    action: str = "answer",
):
    return SimpleNamespace(
        content=json.dumps(
            {
                "response": response,
                "confidence": confidence,
                "action": action,
                "technical_steps": [],
            }
        ),
        tool_calls=[],
    )


class _ScriptedCompletions:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = iter(messages)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self._messages))]
        )


class _ScriptedClient:
    def __init__(self, messages: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(messages))


class _RuleEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _validation(
        tool_name: str,
        arguments: dict[str, Any],
        *,
        approved: bool = True,
        status: str = "approved",
        requires_human_approval: bool = False,
    ) -> ActionValidationResult:
        return ActionValidationResult(
            approved=approved,
            status=status,
            tool_name=tool_name,
            reasons=[
                RuleReason(
                    code="test",
                    passed=approved,
                    message="Scripted validation.",
                )
            ],
            requires_human_approval=requires_human_approval,
            normalized_arguments=arguments,
            budget_used=0,
            budget_remaining=10_000,
        )

    def execute_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ActionExecutionResult:
        self.calls.append((tool_name, arguments))
        if tool_name == "issue_refund" and arguments["amount"] > 1_000:
            validation = self._validation(
                tool_name,
                arguments,
                approved=False,
                status="requires_human_approval",
                requires_human_approval=True,
            )
            return ActionExecutionResult(
                executed=False,
                validation=validation,
            )
        validation = self._validation(tool_name, arguments)
        if tool_name == "get_customer":
            output = {
                "success": True,
                "error": None,
                "customer": {
                    "customer_id": arguments["customer_id"],
                    "plan": "Standard",
                    "email": "private@example.com",
                    "phone": "+91-98765-43210",
                },
            }
        elif tool_name == "search_kb":
            output = {
                "success": True,
                "error": None,
                "chunks": [
                    {
                        "content": "Invoices are billed in advance.",
                        "score": 0.91,
                        "metadata": {
                            "filename": "billing_policy.md",
                            "chunk_id": "billing_policy:0001",
                            "title": "Billing Policy",
                        },
                    }
                ],
            }
        else:
            output = {"success": True, "error": None}
        return ActionExecutionResult(
            executed=True,
            validation=validation,
            tool_output=output,
        )


def _state(message: str, customer_id: str = "CUST0003") -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "sanitized_message": message,
        "input_guardrail": evaluate_input(message),
        "memory_context": MemoryContext(
            conversation_id="conversation-1",
            customer_id=customer_id,
        ),
    }


def test_model_selects_customer_and_kb_tools_then_returns_grounded_answer() -> None:
    client = _ScriptedClient(
        [
            _tool_message(
                "call-1",
                "get_customer",
                {"customer_id": "CUST0003"},
            ),
            _tool_message(
                "call-2",
                "search_kb",
                {"query": "billing cycle and monthly invoice"},
            ),
            _final_message("Your Standard plan is billed in advance."),
        ]
    )
    engine = _RuleEngine()
    handler = LLMToolCallingHandler(
        client=client,
        rule_engine=engine,
        model="test-model",
    )

    result = handler(
        "billing",
        _state("Why is my bill higher this month?"),
    )

    assert [name for name, _ in engine.calls] == ["get_customer", "search_kb"]
    assert [trace.tool for trace in result.tool_calls] == [
        "get_customer",
        "search_kb",
    ]
    assert result.citations == [
        "billing_policy.md#billing_policy:0001"
    ]
    assert result.policy_based
    assert result.action == "answer"
    # Contact fields are redacted before a tool result is sent back to the model.
    second_request_messages = client.chat.completions.requests[1]["messages"]
    tool_payload = json.loads(second_request_messages[-1]["content"])
    assert tool_payload["tool_output"]["customer"]["email"] == "[EMAIL]"
    assert tool_payload["tool_output"]["customer"]["phone"] == "[PHONE]"


def test_cross_customer_tool_request_is_rejected_before_execution() -> None:
    client = _ScriptedClient(
        [
            _tool_message(
                "call-1",
                "get_customer",
                {"customer_id": "CUST9999"},
            ),
            _final_message(
                "I need a human to verify the account.",
                confidence=0.8,
                action="escalate",
            ),
        ]
    )
    engine = _RuleEngine()
    handler = LLMToolCallingHandler(
        client=client,
        rule_engine=engine,
        model="test-model",
    )

    result = handler("billing", _state("Show my account."))

    assert [name for name, _ in engine.calls] == ["escalate_to_human"]
    assert not result.tool_calls[0].executed
    assert result.tool_calls[0].status == "rejected"


def test_large_refund_is_not_executed_and_is_escalated() -> None:
    client = _ScriptedClient(
        [
            _tool_message(
                "call-1",
                "issue_refund",
                {
                    "customer_id": "CUST0094",
                    "amount": 10_000,
                    "reason": "Customer requested refund",
                },
            ),
            _final_message(
                "Your refund request requires human approval.",
                confidence=0.95,
                action="escalate",
            ),
        ]
    )
    engine = _RuleEngine()
    handler = LLMToolCallingHandler(
        client=client,
        rule_engine=engine,
        model="test-model",
    )

    result = handler(
        "refund",
        _state("Refund 10000.", customer_id="CUST0094"),
    )

    assert [name for name, _ in engine.calls] == [
        "issue_refund",
        "escalate_to_human",
    ]
    assert not result.tool_calls[0].executed
    assert result.tool_calls[0].status == "requires_human_approval"
    assert result.status == "escalated"
    assert result.action == "escalate"
