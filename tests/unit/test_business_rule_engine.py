"""Tests for authoritative business-action validation."""

from __future__ import annotations

import json
from pathlib import Path

from customer_ops_agent.business_rules import BusinessRuleEngine
from customer_ops_agent.observability import ObservabilityStore


def test_large_refund_requires_trusted_human_approval() -> None:
    engine = BusinessRuleEngine()
    arguments = {
        "customer_id": "CUST0094",
        "amount": 1_001,
        "reason": "Verified duplicate charge",
    }

    blocked = engine.validate_action("issue_refund", arguments)
    approved = engine.validate_action(
        "issue_refund",
        arguments,
        human_approved=True,
    )

    assert not blocked.approved
    assert blocked.status == "requires_human_approval"
    assert blocked.requires_human_approval
    assert approved.approved
    assert approved.status == "approved"


def test_retention_requires_percentage_tenure_risk_and_unique_offer() -> None:
    engine = BusinessRuleEngine()

    eligible = engine.validate_action(
        "grant_retention_offer",
        {"customer_id": "CUST0058", "pct": 20},
    )
    duplicate = engine.validate_action(
        "grant_retention_offer",
        {"customer_id": "CUST0058", "pct": 20},
    )
    low_risk = engine.validate_action(
        "grant_retention_offer",
        {"customer_id": "CUST0003", "pct": 10},
    )
    excessive = engine.validate_action(
        "grant_retention_offer",
        {"customer_id": "CUST0058", "pct": 21},
    )

    assert eligible.approved
    assert eligible.budget_used == 359.4
    assert not duplicate.approved
    assert any(reason.code == "single_offer_per_run" for reason in duplicate.reasons)
    assert not low_risk.approved
    assert any(
        reason.code == "minimum_churn_risk" and not reason.passed
        for reason in low_risk.reasons
    )
    assert not excessive.approved
    assert excessive.reasons[0].code == "invalid_action_schema"


def test_total_offer_budget_never_exceeds_ten_thousand(tmp_path: Path) -> None:
    customers = [
        {
            "customer_id": f"CUST9{position:03d}",
            "name": f"Eligible Customer {position}",
            "plan": "Business",
            "monthly_fee": 1999,
            "tenure_months": 24,
            "region": "West",
            "email": f"customer{position}@example.com",
            "phone": f"+91-90000-{position:05d}",
            "churn_risk_score": 0.8,
            "outstanding_invoice": 0,
            "entitlements": ["dedicated_support"],
            "active_services": ["streaming"],
            "account_status": "active",
        }
        for position in range(1, 11)
    ]
    customer_path = tmp_path / "customers.json"
    customer_path.write_text(json.dumps(customers), encoding="utf-8")
    engine = BusinessRuleEngine(customer_path)

    decisions = [
        engine.validate_action(
            "grant_retention_offer",
            {"customer_id": customer["customer_id"], "pct": 20},
        )
        for customer in customers
    ]

    assert sum(decision.approved for decision in decisions) == 8
    assert decisions[7].budget_used == 9_595.2
    assert not decisions[8].approved
    assert decisions[8].budget_used <= 10_000
    assert any(
        reason.code == "retention_budget" and not reason.passed
        for reason in decisions[8].reasons
    )


def test_invalid_action_is_rejected_without_tool_execution(tmp_path: Path) -> None:
    engine = BusinessRuleEngine(
        observability_store=ObservabilityStore(
            tmp_path / "tool_calls.csv",
            tmp_path / "conversation_logs.json",
        )
    )

    result = engine.execute_action(
        "issue_refund",
        {
            "customer_id": "CUST0094",
            "amount": 10_000,
            "reason": "Customer requested refund",
        },
    )

    assert not result.executed
    assert result.tool_output is None
    assert result.validation.status == "requires_human_approval"
    assert "requires_human_approval" in (
        tmp_path / "tool_calls.csv"
    ).read_text(encoding="utf-8")
