"""Validation for confidence, grounding, privacy, and response structure."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .input import mask_pii
from .models import (
    AgentOutput,
    ConfidenceResult,
    OutputValidationResult,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.7
_SECRET_PATTERN = re.compile(
    r"\b(?:sk|api|token)[-_][A-Za-z0-9_-]{20,}\b",
    re.IGNORECASE,
)


def check_confidence(
    score: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ConfidenceResult:
    """Convert model confidence into a deterministic continue/escalate decision."""

    accepted = score >= threshold
    return ConfidenceResult(
        accepted=accepted,
        score=score,
        threshold=threshold,
        decision="continue" if accepted else "escalate",
    )


def validate_output(
    candidate: AgentOutput | dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> OutputValidationResult:
    """Validate and sanitize an agent output before it reaches a customer."""

    try:
        output = (
            candidate
            if isinstance(candidate, AgentOutput)
            else AgentOutput.model_validate(candidate)
        )
    except ValidationError as exc:
        return OutputValidationResult(
            valid=False,
            violations=[
                f"Invalid structured output: {error['loc']}: {error['msg']}"
                for error in exc.errors()
            ],
            requires_escalation=True,
        )

    violations: list[str] = []
    requires_escalation = False
    confidence = check_confidence(output.confidence, confidence_threshold)
    if not confidence.accepted:
        violations.append(
            f"Confidence {output.confidence:.2f} is below "
            f"{confidence_threshold:.2f}."
        )
        requires_escalation = True

    if output.policy_based and not output.citations:
        violations.append("Policy-based output requires at least one KB citation.")
        requires_escalation = True

    if any(not citation.strip() for citation in output.citations):
        violations.append("Citations must not contain empty values.")

    if _SECRET_PATTERN.search(output.response):
        violations.append("Potential secret or access token detected in output.")
        requires_escalation = True

    pii = mask_pii(output.response)
    if pii.replacements:
        violations.append("PII was masked in the customer-facing response.")
        output = output.model_copy(update={"response": pii.masked_text})

    return OutputValidationResult(
        valid=not violations,
        output=output,
        violations=violations,
        requires_escalation=requires_escalation,
    )
