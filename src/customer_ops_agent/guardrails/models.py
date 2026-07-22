"""Structured contracts shared by all guardrail components."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class GuardrailDecision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class DetectionResult(BaseModel):
    detected: bool
    score: float = Field(ge=0, le=1)
    categories: list[str] = Field(default_factory=list)


class PIIMaskingResult(BaseModel):
    masked_text: str
    detected_types: list[str] = Field(default_factory=list)
    replacements: int = Field(ge=0)


class InputGuardrailResult(BaseModel):
    decision: GuardrailDecision
    sanitized_text: str
    injection: DetectionResult
    abuse: DetectionResult
    pii: PIIMaskingResult
    reasons: list[str] = Field(default_factory=list)


class ConfidenceResult(BaseModel):
    accepted: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    decision: Literal["continue", "escalate"]


class AgentOutput(BaseModel):
    """Schema expected from response generation before external delivery."""

    response: str = Field(min_length=1, max_length=8_000)
    action: Literal["answer", "tool_call", "escalate", "refuse", "clarify"]
    confidence: float = Field(ge=0, le=1)
    citations: list[str] = Field(default_factory=list)
    policy_based: bool = False


class OutputValidationResult(BaseModel):
    valid: bool
    output: AgentOutput | None = None
    violations: list[str] = Field(default_factory=list)
    requires_escalation: bool = False


class ToolValidationResult(BaseModel):
    valid: bool
    tool_name: str
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolCallLimitResult(BaseModel):
    allowed: bool
    calls_used: int = Field(ge=0)
    calls_remaining: int = Field(ge=0)
    error: str | None = None


class DeadLoopResult(BaseModel):
    dead_loop_detected: bool
    signature: str
    occurrences: int = Field(ge=0)
    error: str | None = None


class RetryResult(BaseModel):
    success: bool
    attempts: int = Field(ge=1)
    value: Any = None
    error: str | None = None


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerResult(BaseModel):
    success: bool
    state: CircuitState
    value: Any = None
    error: str | None = None
    rejected: bool = False
