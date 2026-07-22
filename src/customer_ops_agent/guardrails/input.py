"""Deterministic input screening and privacy-preserving masking."""

from __future__ import annotations

import re

from .models import (
    DetectionResult,
    GuardrailDecision,
    InputGuardrailResult,
    PIIMaskingResult,
)


# Labels are returned instead of raw matches so audit records do not repeat
# hostile instructions or sensitive customer text.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|forget|disregard)\b.{0,40}\b"
            r"(previous|prior|system|developer|instructions?|rules?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.95,
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|expose)\b.{0,40}\b"
            r"(system prompt|internal (?:prompt|policy|instructions?)|api key)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.95,
    ),
    (
        "role_impersonation",
        re.compile(
            r"\b(?:act|treat this|you are now)\b.{0,40}\b"
            r"(?:administrator|developer|system|root|manager)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.85,
    ),
    (
        "control_bypass",
        re.compile(
            r"\b(bypass|disable|override)\b.{0,35}\b"
            r"(guardrails?|safety|approval|policy|rules?|limits?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.9,
    ),
)

_ABUSE_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "violent_threat",
        re.compile(
            r"\b(kill|hurt|attack|bomb|shoot|destroy)\b.{0,35}\b"
            r"(you|staff|team|office|store|employee)",
            re.IGNORECASE | re.DOTALL,
        ),
        0.98,
    ),
    (
        "abusive_language",
        re.compile(
            r"\b(idiots?|morons?|stupid|useless|incompetent|damn|trash)\b",
            re.IGNORECASE,
        ),
        0.6,
    ),
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])"),
        "[EMAIL]",
    ),
    (
        "payment_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        "[PAYMENT_CARD]",
    ),
    (
        "aadhaar",
        re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
        "[AADHAAR]",
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{4}[- ]?\d{5}(?!\d)"),
        "[PHONE]",
    ),
)


def detect_prompt_injection(text: str) -> DetectionResult:
    """Detect high-precision instruction override and exfiltration patterns."""

    matches = [
        (label, weight)
        for label, pattern, weight in _INJECTION_PATTERNS
        if pattern.search(text)
    ]
    return DetectionResult(
        detected=bool(matches),
        score=max((weight for _, weight in matches), default=0),
        categories=[label for label, _ in matches],
    )


def detect_abuse(text: str) -> DetectionResult:
    """Separate ordinary abusive language from high-risk violent threats."""

    matches = [
        (label, weight)
        for label, pattern, weight in _ABUSE_PATTERNS
        if pattern.search(text)
    ]
    return DetectionResult(
        detected=bool(matches),
        score=max((weight for _, weight in matches), default=0),
        categories=[label for label, _ in matches],
    )


def mask_pii(text: str) -> PIIMaskingResult:
    """Mask common customer PII while retaining enough text for intent routing."""

    masked = text
    detected_types: list[str] = []
    replacement_count = 0
    for pii_type, pattern, replacement in _PII_PATTERNS:
        masked, count = pattern.subn(replacement, masked)
        if count:
            detected_types.append(pii_type)
            replacement_count += count

    return PIIMaskingResult(
        masked_text=masked,
        detected_types=detected_types,
        replacements=replacement_count,
    )


def evaluate_input(text: str) -> InputGuardrailResult:
    """Run all input checks and return one explicit routing decision."""

    injection = detect_prompt_injection(text)
    abuse = detect_abuse(text)
    pii = mask_pii(text)
    reasons: list[str] = []

    if injection.detected:
        reasons.append("Prompt-injection pattern detected.")
    if "violent_threat" in abuse.categories:
        reasons.append("Violent threat requires immediate human handling.")
    elif abuse.detected:
        reasons.append("Abusive language requires a respectful boundary.")
    if pii.replacements:
        reasons.append("PII was masked before model processing.")

    if injection.score >= 0.8 or "violent_threat" in abuse.categories:
        decision = GuardrailDecision.BLOCK
    elif abuse.detected:
        decision = GuardrailDecision.REVIEW
    else:
        decision = GuardrailDecision.ALLOW

    return InputGuardrailResult(
        decision=decision,
        sanitized_text=pii.masked_text,
        injection=injection,
        abuse=abuse,
        pii=pii,
        reasons=reasons,
    )
