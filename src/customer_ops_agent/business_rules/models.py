"""Structured decisions returned by the business rule engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RuleReason(BaseModel):
    """One auditable rule evaluation."""

    code: str
    passed: bool
    message: str


class ActionValidationResult(BaseModel):
    """Final allow, reject, or approval decision for one requested action."""

    approved: bool
    status: Literal["approved", "rejected", "requires_human_approval"]
    tool_name: str
    reasons: list[RuleReason] = Field(default_factory=list)
    requires_human_approval: bool = False
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    budget_used: float
    budget_remaining: float


class ActionExecutionResult(BaseModel):
    """Validation and optional tool output returned as one structured boundary."""

    executed: bool
    validation: ActionValidationResult
    tool_output: dict[str, Any] | None = None
    error: str | None = None
