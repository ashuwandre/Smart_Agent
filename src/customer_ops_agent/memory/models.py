"""Pydantic contracts for JSON-backed conversation memory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Interaction(BaseModel):
    interaction_id: str
    role: Literal["customer", "assistant", "system"]
    message: str = Field(min_length=1, max_length=8_000)
    timestamp: datetime
    include_in_context: bool = True


class OfferMemory(BaseModel):
    percentage: float = Field(gt=0, le=100)
    status: str = Field(min_length=1, max_length=100)
    recorded_at: datetime


class RefundMemory(BaseModel):
    amount: float = Field(gt=0)
    status: str = Field(min_length=1, max_length=100)
    recorded_at: datetime


class ConversationMemory(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=200)
    customer_id: str | None = Field(default=None, pattern=r"^CUST\d{4}$")
    interactions: list[Interaction] = Field(default_factory=list, max_length=10)
    offer_given: OfferMemory | None = None
    refund_requested: RefundMemory | None = None
    technical_steps: list[str] = Field(default_factory=list, max_length=10)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryContext(BaseModel):
    """Safe context supplied to orchestration for the next turn."""

    conversation_id: str
    customer_id: str | None = None
    previous_messages: list[Interaction] = Field(default_factory=list)
    offer_given: OfferMemory | None = None
    refund_requested: RefundMemory | None = None
    technical_steps: list[str] = Field(default_factory=list)


class MemoryDocument(BaseModel):
    """Versioned root object keeps future JSON migrations explicit."""

    version: int = 1
    conversations: dict[str, ConversationMemory] = Field(default_factory=dict)
