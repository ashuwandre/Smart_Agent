"""Fail-closed validation and execution for every customer action."""

from __future__ import annotations

import json
from decimal import Decimal
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from customer_ops_agent.guardrails import validate_tool_call
from customer_ops_agent.observability import (
    DEFAULT_OBSERVABILITY_STORE,
    ObservabilityStore,
    ToolCallEvent,
)
from customer_ops_agent.tools import (
    create_ticket,
    escalate_to_human,
    get_customer,
    grant_retention_offer,
    issue_refund,
    search_kb,
)
from customer_ops_agent.tools.models import CustomerRecord

from .models import ActionExecutionResult, ActionValidationResult, RuleReason


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "synthetic" / "customers.json"

REFUND_APPROVAL_THRESHOLD = Decimal("1000")
MAXIMUM_OFFER_PERCENTAGE = Decimal("20")
RETENTION_BUDGET = Decimal("10000")
MINIMUM_TENURE_MONTHS = 12
MINIMUM_CHURN_RISK = Decimal("0.70")
OFFER_DURATION_MONTHS = 3

_TOOLS = {
    "get_customer": get_customer,
    "search_kb": search_kb,
    "grant_retention_offer": grant_retention_offer,
    "issue_refund": issue_refund,
    "create_ticket": create_ticket,
    "escalate_to_human": escalate_to_human,
}


class BusinessRuleEngine:
    """Validate trusted customer facts and reserve financial budget atomically."""

    def __init__(
        self,
        customers_path: str | Path = CUSTOMERS_PATH,
        observability_store: ObservabilityStore | None = None,
    ) -> None:
        raw_customers = json.loads(Path(customers_path).read_text(encoding="utf-8"))
        customers = [
            CustomerRecord.model_validate(customer) for customer in raw_customers
        ]
        self._customers = {customer.customer_id: customer for customer in customers}
        self._budget_used = Decimal("0")
        self._offer_customers: set[str] = set()
        self._lock = RLock()
        self._observability_store = (
            observability_store or DEFAULT_OBSERVABILITY_STORE
        )

    @staticmethod
    def _money(value: Decimal) -> float:
        return float(value.quantize(Decimal("0.01")))

    def _budget_snapshot(self) -> tuple[float, float]:
        with self._lock:
            return (
                self._money(self._budget_used),
                self._money(RETENTION_BUDGET - self._budget_used),
            )

    def _result(
        self,
        *,
        approved: bool,
        status: str,
        tool_name: str,
        reasons: list[RuleReason],
        normalized_arguments: dict[str, Any],
        requires_human_approval: bool = False,
    ) -> ActionValidationResult:
        used, remaining = self._budget_snapshot()
        return ActionValidationResult(
            approved=approved,
            status=status,
            tool_name=tool_name,
            reasons=reasons,
            requires_human_approval=requires_human_approval,
            normalized_arguments=normalized_arguments,
            budget_used=used,
            budget_remaining=remaining,
        )

    def validate_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        human_approved: bool = False,
        reserve_budget: bool = True,
    ) -> ActionValidationResult:
        """Validate every action and reserve offer budget only after all rules pass."""

        schema_result = validate_tool_call(tool_name, arguments)
        if not schema_result.valid:
            return self._result(
                approved=False,
                status="rejected",
                tool_name=tool_name,
                reasons=[
                    RuleReason(
                        code="invalid_action_schema",
                        passed=False,
                        message=schema_result.error or "Action validation failed.",
                    )
                ],
                normalized_arguments={},
            )

        normalized = schema_result.normalized_arguments
        reasons = [
            RuleReason(
                code="valid_action_schema",
                passed=True,
                message="Tool and arguments passed strict validation.",
            )
        ]

        if tool_name == "issue_refund":
            amount = Decimal(str(normalized["amount"]))
            requires_approval = amount > REFUND_APPROVAL_THRESHOLD
            reasons.append(
                RuleReason(
                    code="refund_human_approval",
                    passed=not requires_approval or human_approved,
                    message=(
                        "Refund above ₹1,000 has verified human approval."
                        if requires_approval and human_approved
                        else "Refund above ₹1,000 requires human approval."
                        if requires_approval
                        else "Refund is within the autonomous ₹1,000 threshold."
                    ),
                )
            )
            if requires_approval and not human_approved:
                return self._result(
                    approved=False,
                    status="requires_human_approval",
                    tool_name=tool_name,
                    reasons=reasons,
                    normalized_arguments=normalized,
                    requires_human_approval=True,
                )

        if tool_name == "grant_retention_offer":
            return self._validate_retention_offer(
                tool_name,
                normalized,
                reasons,
                reserve_budget,
            )

        reasons.append(
            RuleReason(
                code="action_allowed",
                passed=True,
                message="No additional business restriction rejected this action.",
            )
        )
        return self._result(
            approved=True,
            status="approved",
            tool_name=tool_name,
            reasons=reasons,
            normalized_arguments=normalized,
        )

    def _validate_retention_offer(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reasons: list[RuleReason],
        reserve_budget: bool,
    ) -> ActionValidationResult:
        customer_id = str(arguments["customer_id"])
        customer = self._customers.get(customer_id)
        if customer is None:
            reasons.append(
                RuleReason(
                    code="customer_exists",
                    passed=False,
                    message=f"Customer {customer_id} was not found.",
                )
            )
            return self._result(
                approved=False,
                status="rejected",
                tool_name=tool_name,
                reasons=reasons,
                normalized_arguments=arguments,
            )

        percentage = Decimal(str(arguments["pct"]))
        offer_cost = (
            Decimal(str(customer.monthly_fee))
            * percentage
            / Decimal("100")
            * OFFER_DURATION_MONTHS
        ).quantize(Decimal("0.01"))

        checks = [
            RuleReason(
                code="offer_percentage",
                passed=percentage <= MAXIMUM_OFFER_PERCENTAGE,
                message=(
                    f"Offer must not exceed {MAXIMUM_OFFER_PERCENTAGE}%."
                ),
            ),
            RuleReason(
                code="minimum_tenure",
                passed=customer.tenure_months >= MINIMUM_TENURE_MONTHS,
                message=(
                    f"Tenure is {customer.tenure_months} months; "
                    f"minimum is {MINIMUM_TENURE_MONTHS}."
                ),
            ),
            RuleReason(
                code="minimum_churn_risk",
                passed=Decimal(str(customer.churn_risk_score))
                >= MINIMUM_CHURN_RISK,
                message=(
                    f"Churn risk is {customer.churn_risk_score:.2f}; "
                    f"minimum is {MINIMUM_CHURN_RISK}."
                ),
            ),
            RuleReason(
                code="active_account",
                passed=customer.account_status == "active",
                message="Retention offers require an active account.",
            ),
            RuleReason(
                code="no_outstanding_invoice",
                passed=customer.outstanding_invoice == 0,
                message="Retention offers require no outstanding invoice.",
            ),
        ]
        reasons.extend(checks)

        with self._lock:
            duplicate = customer_id in self._offer_customers
            within_budget = self._budget_used + offer_cost <= RETENTION_BUDGET
            reasons.extend(
                [
                    RuleReason(
                        code="single_offer_per_run",
                        passed=not duplicate,
                        message="Customer must not receive duplicate offers in one run.",
                    ),
                    RuleReason(
                        code="retention_budget",
                        passed=within_budget,
                        message=(
                            f"Offer costs ₹{offer_cost}; total run budget is "
                            f"₹{RETENTION_BUDGET}."
                        ),
                    ),
                ]
            )
            approved = all(reason.passed for reason in reasons)
            if approved and reserve_budget:
                # Reserving during validation prevents concurrent requests from
                # both observing the same remaining budget.
                self._budget_used += offer_cost
                self._offer_customers.add(customer_id)

        return self._result(
            approved=approved,
            status="approved" if approved else "rejected",
            tool_name=tool_name,
            reasons=reasons,
            normalized_arguments=arguments,
        )

    def execute_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        human_approved: bool = False,
    ) -> ActionExecutionResult:
        """Reject invalid requests before dispatching to an existing mock tool."""

        started_at = perf_counter()
        validation = self.validate_action(
            tool_name,
            arguments,
            human_approved=human_approved,
            reserve_budget=True,
        )
        if not validation.approved:
            failed_reasons = [
                reason.message for reason in validation.reasons if not reason.passed
            ]
            self._observability_store.log_tool_call(
                ToolCallEvent(
                    timestamp=datetime.now(UTC),
                    customer=arguments.get("customer_id"),
                    tool=tool_name,
                    latency=round((perf_counter() - started_at) * 1_000, 3),
                    status=validation.status,
                    reason="; ".join(failed_reasons),
                )
            )
            return ActionExecutionResult(executed=False, validation=validation)

        tool = _TOOLS[tool_name]
        call_arguments = dict(validation.normalized_arguments)
        if tool_name == "issue_refund" and human_approved:
            call_arguments["human_approved"] = True

        try:
            output: BaseModel = tool(**call_arguments)
        except Exception as exc:
            return ActionExecutionResult(
                executed=False,
                validation=validation,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ActionExecutionResult(
            executed=True,
            validation=validation,
            tool_output=output.model_dump(mode="json"),
        )
