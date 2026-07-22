"""Structured specialist results passed between agent graph nodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Route = Literal["billing", "refund", "technical", "retention"]
ResponseAction = Literal["answer", "escalate", "clarify", "refuse"]


class ToolCallTrace(BaseModel):
    """Observable record of one model-selected tool request."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    executed: bool
    status: str
    output: dict[str, Any] | None = None
    reason: str = ""


class HandlerResult(BaseModel):
    """Validated output from a specialist tool-calling loop."""

    handler: Route
    status: Literal["completed", "escalated", "clarification", "failed"]
    summary: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0, le=1)
    action: ResponseAction
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    technical_steps: list[str] = Field(default_factory=list)
    policy_based: bool = False


class SpecialistDraft(BaseModel):
    """Strict final JSON requested from the specialist LLM."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0, le=1)
    action: ResponseAction
    technical_steps: list[str]
