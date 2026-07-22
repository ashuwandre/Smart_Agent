"""Structured telemetry contracts for tools and full conversations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ToolCallEvent(BaseModel):
    """CSV columns required for one observable tool execution."""

    timestamp: datetime
    customer: str | None = None
    tool: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency: float = Field(ge=0)
    retry: int = Field(default=0, ge=0)
    status: str
    loop_count: int = Field(default=0, ge=0)
    reason: str = ""


class ConversationLog(BaseModel):
    """Privacy-safe end-to-end request telemetry."""

    timestamp: datetime
    conversation_id: str
    customer: str | None = None
    request: str
    response: str | None = None
    route: str | None = None
    status: Literal["completed", "blocked", "error"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency: float = Field(ge=0)
    guardrail_decision: str | None = None
    error: str | None = None
    circuit_breakers: dict[str, str] = Field(default_factory=dict)


class ConversationLogDocument(BaseModel):
    version: int = 1
    requests: list[ConversationLog] = Field(default_factory=list)
