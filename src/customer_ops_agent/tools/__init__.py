"""Public mock-tool interface; no agent or orchestration is defined here."""

from .audit import tool_telemetry_context
from .mock_tools import (
    create_ticket,
    escalate_to_human,
    get_customer,
    grant_retention_offer,
    issue_refund,
    search_kb,
)
from .models import (
    EscalationResult,
    GetCustomerResult,
    RefundResult,
    RetentionOfferResult,
    SearchKBResult,
    TicketResult,
)

__all__ = [
    "EscalationResult",
    "GetCustomerResult",
    "RefundResult",
    "RetentionOfferResult",
    "SearchKBResult",
    "TicketResult",
    "create_ticket",
    "escalate_to_human",
    "get_customer",
    "grant_retention_offer",
    "issue_refund",
    "search_kb",
    "tool_telemetry_context",
]
