"""Pydantic contracts returned by the deterministic mock tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Common fields let orchestration inspect failures without parsing prose."""

    success: bool
    error: str | None = None


class CustomerRecord(BaseModel):
    """Validated representation of a synthetic customer record."""

    customer_id: str
    name: str
    plan: str
    monthly_fee: float
    tenure_months: int
    region: str
    email: str
    phone: str
    churn_risk_score: float = Field(ge=0, le=1)
    outstanding_invoice: Literal[0, 1]
    entitlements: list[str]
    active_services: list[str]
    account_status: str


class GetCustomerResult(ToolResult):
    """Result of a customer lookup."""

    customer: CustomerRecord | None = None


class KBChunkMetadata(BaseModel):
    """Source fields required for grounded citations."""

    filename: str
    chunk_id: str
    title: str


class KBChunk(BaseModel):
    """One semantic search match."""

    content: str
    score: float
    metadata: KBChunkMetadata


class SearchKBResult(ToolResult):
    """Top knowledge-base matches for a query."""

    chunks: list[KBChunk] = Field(default_factory=list)


class RetentionOfferResult(ToolResult):
    """Outcome of an eligibility, cap, and budget checked offer request."""

    status: Literal["granted", "denied"]
    customer_id: str
    percentage: float
    duration_months: int = 3
    offer_cost: float = 0
    remaining_budget: float
    reason: str


class RefundResult(ToolResult):
    """Outcome of a refund request or approval gate."""

    status: Literal["issued", "pending_approval", "denied"]
    customer_id: str
    amount: float
    approval_required: bool
    transaction_id: str | None = None
    reason: str


class TicketResult(ToolResult):
    """Created support ticket details."""

    status: Literal["created", "denied"]
    customer_id: str
    ticket_id: str | None = None
    summary: str


class EscalationResult(ToolResult):
    """Human handoff details."""

    status: Literal["escalated", "denied"]
    reason: str
    customer_id: str | None = None
    escalation_id: str | None = None
