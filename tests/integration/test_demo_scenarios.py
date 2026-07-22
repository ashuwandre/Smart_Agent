"""End-to-end scenario tests driven by the versioned demo dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from customer_ops_agent.agent import MockPlanner, MockSpecialistHandler, build_graph
from customer_ops_agent.business_rules import BusinessRuleEngine
from customer_ops_agent.guardrails import CircuitBreaker, DeadLoopDetector
from customer_ops_agent.memory import ConversationMemoryStore
from customer_ops_agent.observability import ObservabilityStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = PROJECT_ROOT / "data" / "synthetic" / "demo_scenarios.json"


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict[str, Any]]:
    records = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {record["category"]: record for record in records}


def _graph(tmp_path: Path):
    return build_graph(
        planner=MockPlanner(),
        specialist_handler=MockSpecialistHandler(),
        memory_store=ConversationMemoryStore(tmp_path / "memory.json"),
        observability_store=ObservabilityStore(
            tmp_path / "tool_calls.csv",
            tmp_path / "conversation_logs.json",
        ),
    )


def test_demo_dataset_contains_all_required_scenarios(
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert set(scenarios) == {
        "billing",
        "refund",
        "cancellation",
        "prompt_injection",
        "abusive_user",
        "ambiguous_query",
        "technical_issue",
        "budget_exhausted",
        "refund_approval",
        "circuit_breaker",
        "dead_loop",
    }


@pytest.mark.parametrize(
    "category",
    ["billing", "refund", "cancellation", "technical_issue"],
)
def test_standard_requests_reach_expected_handler(
    category: str,
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios[category]

    result = _graph(tmp_path).invoke(scenario["input"])

    assert result["route"] == scenario["expected"]["route"]
    assert result["handler_result"].handler == scenario["expected"]["route"]
    assert (
        result["input_guardrail"].decision
        == scenario["expected"]["guardrail_decision"]
    )


def test_prompt_injection_never_reaches_handler(
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios["prompt_injection"]

    result = _graph(tmp_path).invoke(scenario["input"])

    assert result["input_guardrail"].decision == "block"
    assert result["output_candidate"].action == "refuse"
    assert "handler_result" not in result


def test_abusive_user_is_reviewed_but_request_remains_actionable(
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios["abusive_user"]

    result = _graph(tmp_path).invoke(scenario["input"])

    assert result["input_guardrail"].decision == "review"
    assert result["route"] == "billing"


def test_ambiguous_query_executes_no_customer_action(
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios["ambiguous_query"]

    result = _graph(tmp_path).invoke(scenario["input"])

    assert not scenario["expected"]["customer_action_executed"]
    assert "No customer action was executed" in result["response"]


def test_budget_exhaustion_rejects_ninth_offer(
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios["budget_exhausted"]
    monthly_fee = scenario["input"]["monthly_fee"]
    customer_count = scenario["input"]["prior_approved_offers"] + 1
    customers = [
        {
            "customer_id": f"CUST8{position:03d}",
            "name": f"Budget Customer {position}",
            "plan": "Business",
            "monthly_fee": monthly_fee,
            "tenure_months": 24,
            "region": "West",
            "email": f"budget{position}@example.com",
            "phone": f"+91-80000-{position:05d}",
            "churn_risk_score": 0.8,
            "outstanding_invoice": 0,
            "entitlements": ["dedicated_support"],
            "active_services": ["streaming"],
            "account_status": "active",
        }
        for position in range(1, customer_count + 1)
    ]
    customer_path = tmp_path / "customers.json"
    customer_path.write_text(json.dumps(customers), encoding="utf-8")
    engine = BusinessRuleEngine(customer_path)

    decisions = [
        engine.validate_action(
            "grant_retention_offer",
            {
                "customer_id": customer["customer_id"],
                "pct": scenario["input"]["offer_percentage"],
            },
        )
        for customer in customers
    ]

    assert all(decision.approved for decision in decisions[:-1])
    assert not decisions[-1].approved
    assert any(
        reason.code == scenario["expected"]["reason_code"] and not reason.passed
        for reason in decisions[-1].reasons
    )


def test_large_refund_enters_human_approval_flow(
    scenarios: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    scenario = scenarios["refund_approval"]
    engine = BusinessRuleEngine(
        observability_store=ObservabilityStore(
            tmp_path / "tool_calls.csv",
            tmp_path / "conversation_logs.json",
        )
    )

    result = engine.execute_action("issue_refund", scenario["input"])

    assert result.executed == scenario["expected"]["executed"]
    assert result.validation.status == scenario["expected"]["status"]


def test_circuit_breaker_opens_and_rejects_next_call(
    scenarios: dict[str, dict[str, Any]],
) -> None:
    scenario = scenarios["circuit_breaker"]
    breaker = CircuitBreaker(
        failure_threshold=scenario["input"]["failure_threshold"],
        recovery_timeout_seconds=scenario["input"][
            "recovery_timeout_seconds"
        ],
        name="demo-dependency",
    )

    def unavailable() -> None:
        raise ConnectionError("demo dependency unavailable")

    for _ in range(scenario["input"]["failure_threshold"]):
        breaker.call(unavailable)
    rejected = breaker.call(lambda: "must not execute")

    assert breaker.state == scenario["expected"]["state"]
    assert rejected.rejected == scenario["expected"]["next_call_rejected"]


def test_dead_loop_is_detected_at_configured_repetition(
    scenarios: dict[str, dict[str, Any]],
) -> None:
    scenario = scenarios["dead_loop"]
    detector = DeadLoopDetector(
        maximum_occurrences=scenario["input"]["repetitions"],
        window_size=10,
    )

    results = [
        detector.record(
            scenario["input"]["node"],
            scenario["input"]["tool"],
            scenario["input"]["arguments"],
        )
        for _ in range(scenario["input"]["repetitions"])
    ]

    assert results[-1].dead_loop_detected
    assert results[-1].dead_loop_detected == scenario["expected"][
        "dead_loop_detected"
    ]
