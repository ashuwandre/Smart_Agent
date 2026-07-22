"""Deterministic local tools for the simulated customer-operations environment."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock

from customer_ops_agent.rag import build_index as rag_build_index
from customer_ops_agent.rag import search as rag_search

from .audit import audit_tool
from .models import (
    CustomerRecord,
    EscalationResult,
    GetCustomerResult,
    KBChunk,
    RefundResult,
    RetentionOfferResult,
    SearchKBResult,
    TicketResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "synthetic" / "customers.json"

# These values mirror the versioned KB. The tool enforces them in code because
# safety-critical limits must not depend on an LLM following prose.
RETENTION_MAX_PERCENTAGE = Decimal("20")
RETENTION_DURATION_MONTHS = 3
RETENTION_RUN_BUDGET = Decimal("10000")
REFUND_APPROVAL_THRESHOLD = Decimal("1000")

_state_lock = RLock()
_customers: dict[str, CustomerRecord] | None = None
_retention_budget_used = Decimal("0")
_retention_customers: set[str] = set()
_refund_sequence = 0
_ticket_sequence = 0
_escalation_sequence = 0
_rag_ready = False


def _load_customers() -> dict[str, CustomerRecord]:
    """Load and validate synthetic records once; the source data is immutable."""

    global _customers
    with _state_lock:
        if _customers is None:
            raw_records = json.loads(CUSTOMERS_PATH.read_text(encoding="utf-8"))
            validated = [CustomerRecord.model_validate(record) for record in raw_records]
            _customers = {record.customer_id: record for record in validated}
        return _customers


def _find_customer(customer_id: str) -> CustomerRecord | None:
    """Normalize lookup keys without changing the stored identifier."""

    return _load_customers().get(customer_id.strip().upper())


def _money(value: Decimal) -> float:
    """Expose money as a two-decimal JSON number while calculating with Decimal."""

    return float(value.quantize(Decimal("0.01")))


def _remaining_retention_budget() -> Decimal:
    return RETENTION_RUN_BUDGET - _retention_budget_used


@audit_tool
def get_customer(customer_id: str) -> GetCustomerResult:
    """Return one validated customer record."""

    clean_id = customer_id.strip().upper()
    if not clean_id:
        return GetCustomerResult(success=False, error="customer_id must not be empty")

    customer = _find_customer(clean_id)
    if customer is None:
        return GetCustomerResult(
            success=False,
            error=f"Customer {clean_id} was not found",
        )
    return GetCustomerResult(success=True, customer=customer)


@audit_tool
def search_kb(query: str) -> SearchKBResult:
    """Return the top three grounded KB chunks from the RAG module."""

    clean_query = query.strip()
    if not clean_query:
        return SearchKBResult(success=False, error="query must not be empty")

    global _rag_ready
    try:
        # Index construction is guarded so concurrent first requests pay for a
        # single embedding build rather than creating duplicate API calls.
        with _state_lock:
            if not _rag_ready:
                rag_build_index()
                _rag_ready = True
        matches = rag_search(clean_query)
    except Exception as exc:
        return SearchKBResult(
            success=False,
            error=f"Knowledge-base search failed: {type(exc).__name__}: {exc}",
        )

    return SearchKBResult(
        success=True,
        chunks=[KBChunk.model_validate(match) for match in matches],
    )


@audit_tool
def grant_retention_offer(
    customer_id: str,
    pct: float,
) -> RetentionOfferResult:
    """Grant an eligible offer while atomically enforcing caps and run budget."""

    global _retention_budget_used

    clean_id = customer_id.strip().upper()
    customer = _find_customer(clean_id)
    with _state_lock:
        remaining = _remaining_retention_budget()

    if customer is None:
        return RetentionOfferResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            percentage=pct,
            remaining_budget=_money(remaining),
            reason="Customer was not found.",
            error=f"Customer {clean_id} was not found",
        )

    try:
        percentage = Decimal(str(pct))
    except InvalidOperation:
        percentage = Decimal("-1")

    denial_reason: str | None = None
    if percentage <= 0:
        denial_reason = "Offer percentage must be greater than zero."
    elif customer.account_status != "active":
        denial_reason = "Retention offers require an active account."
    elif customer.outstanding_invoice == 1:
        denial_reason = "Outstanding invoices make the customer ineligible."
    elif customer.tenure_months < 12:
        denial_reason = "Customer tenure must be at least 12 months."
    elif customer.churn_risk_score < 0.70:
        denial_reason = "Churn risk must be at least 0.70."
    elif percentage > RETENTION_MAX_PERCENTAGE:
        denial_reason = (
            f"Requested {percentage}% exceeds the maximum offer of "
            f"{RETENTION_MAX_PERCENTAGE}%."
        )

    offer_cost = (
        Decimal(str(customer.monthly_fee))
        * percentage
        / Decimal("100")
        * RETENTION_DURATION_MONTHS
        if percentage > 0
        else Decimal("0")
    ).quantize(Decimal("0.01"))

    with _state_lock:
        if denial_reason is None and clean_id in _retention_customers:
            denial_reason = "A retention offer was already granted in this run."
        if (
            denial_reason is None
            and _retention_budget_used + offer_cost > RETENTION_RUN_BUDGET
        ):
            denial_reason = "The remaining run budget is insufficient."

        if denial_reason is None:
            _retention_budget_used += offer_cost
            _retention_customers.add(clean_id)
        remaining = _remaining_retention_budget()

    if denial_reason is not None:
        return RetentionOfferResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            percentage=float(percentage),
            offer_cost=_money(offer_cost),
            remaining_budget=_money(remaining),
            reason=denial_reason,
        )

    return RetentionOfferResult(
        success=True,
        status="granted",
        customer_id=clean_id,
        percentage=float(percentage),
        offer_cost=_money(offer_cost),
        remaining_budget=_money(remaining),
        reason="Customer is eligible and the offer is within cap and run budget.",
    )


@audit_tool
def issue_refund(
    customer_id: str,
    amount: float,
    reason: str = "Customer requested refund",
    human_approved: bool = False,
) -> RefundResult:
    """Issue a refund or return a non-executing approval requirement."""

    clean_id = customer_id.strip().upper()
    customer = _find_customer(clean_id)
    try:
        refund_amount = Decimal(str(amount))
    except InvalidOperation:
        refund_amount = Decimal("-1")

    if customer is None:
        return RefundResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            amount=amount,
            approval_required=False,
            reason="Customer was not found.",
            error=f"Customer {clean_id} was not found",
        )
    if refund_amount <= 0:
        return RefundResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            amount=amount,
            approval_required=False,
            reason="Refund amount must be greater than zero.",
        )
    if not reason.strip():
        return RefundResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            amount=_money(refund_amount),
            approval_required=False,
            reason="A refund reason is required.",
        )

    approval_required = refund_amount > REFUND_APPROVAL_THRESHOLD
    if approval_required and not human_approved:
        return RefundResult(
            success=True,
            status="pending_approval",
            customer_id=clean_id,
            amount=_money(refund_amount),
            approval_required=True,
            reason="Refund exceeds ₹1,000 and was not executed.",
        )

    global _refund_sequence
    with _state_lock:
        _refund_sequence += 1
        transaction_id = f"RFND{_refund_sequence:06d}"

    return RefundResult(
        success=True,
        status="issued",
        customer_id=clean_id,
        amount=_money(refund_amount),
        approval_required=approval_required,
        transaction_id=transaction_id,
        reason=reason.strip(),
    )


@audit_tool
def create_ticket(customer_id: str, summary: str) -> TicketResult:
    """Create a deterministic local support ticket."""

    clean_id = customer_id.strip().upper()
    if _find_customer(clean_id) is None:
        return TicketResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            summary=summary.strip(),
            error=f"Customer {clean_id} was not found",
        )
    if not summary.strip():
        return TicketResult(
            success=False,
            status="denied",
            customer_id=clean_id,
            summary="",
            error="summary must not be empty",
        )

    global _ticket_sequence
    with _state_lock:
        _ticket_sequence += 1
        ticket_id = f"TKT{_ticket_sequence:06d}"

    return TicketResult(
        success=True,
        status="created",
        customer_id=clean_id,
        ticket_id=ticket_id,
        summary=summary.strip(),
    )


@audit_tool
def escalate_to_human(
    reason: str,
    customer_id: str | None = None,
) -> EscalationResult:
    """Create a terminal human-handoff record for later orchestration."""

    clean_reason = reason.strip()
    clean_id = customer_id.strip().upper() if customer_id else None
    if not clean_reason:
        return EscalationResult(
            success=False,
            status="denied",
            reason="Escalation reason must not be empty.",
            customer_id=clean_id,
            error="reason must not be empty",
        )
    if clean_id and _find_customer(clean_id) is None:
        return EscalationResult(
            success=False,
            status="denied",
            reason=clean_reason,
            customer_id=clean_id,
            error=f"Customer {clean_id} was not found",
        )

    global _escalation_sequence
    with _state_lock:
        _escalation_sequence += 1
        escalation_id = f"ESC{_escalation_sequence:06d}"

    return EscalationResult(
        success=True,
        status="escalated",
        reason=clean_reason,
        customer_id=clean_id,
        escalation_id=escalation_id,
    )


def _reset_mock_state() -> None:
    """Reset in-memory state for isolated tests; this is not a public tool."""

    global _customers
    global _escalation_sequence
    global _rag_ready
    global _refund_sequence
    global _retention_budget_used
    global _ticket_sequence

    with _state_lock:
        _customers = None
        _retention_budget_used = Decimal("0")
        _retention_customers.clear()
        _refund_sequence = 0
        _ticket_sequence = 0
        _escalation_sequence = 0
        _rag_ready = False
