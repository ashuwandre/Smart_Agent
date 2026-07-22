"""Allowlist and argument validation for calls into existing mock tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import ToolValidationResult


class _StrictArguments(BaseModel):
    """Reject unknown fields so models cannot smuggle privileged parameters."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _CustomerArguments(_StrictArguments):
    customer_id: str = Field(pattern=r"^CUST\d{4}$")


class _SearchKBArguments(_StrictArguments):
    query: str = Field(min_length=2, max_length=1_000)


class _RetentionArguments(_CustomerArguments):
    pct: float = Field(gt=0, le=20)


class _RefundArguments(_CustomerArguments):
    amount: float = Field(gt=0, le=1_000_000)
    reason: str = Field(
        default="Customer requested refund",
        min_length=3,
        max_length=500,
    )


class _TicketArguments(_CustomerArguments):
    summary: str = Field(min_length=3, max_length=1_000)


class _EscalationArguments(_StrictArguments):
    reason: str = Field(min_length=3, max_length=1_000)
    customer_id: str | None = Field(
        default=None,
        pattern=r"^CUST\d{4}$",
    )


# The refund schema deliberately excludes ``human_approved``. An LLM must not
# be able to claim human authority even though the underlying tool supports a
# trusted approval workflow.
_TOOL_ARGUMENT_MODELS: dict[str, type[_StrictArguments]] = {
    "get_customer": _CustomerArguments,
    "search_kb": _SearchKBArguments,
    "grant_retention_offer": _RetentionArguments,
    "issue_refund": _RefundArguments,
    "create_ticket": _TicketArguments,
    "escalate_to_human": _EscalationArguments,
}


def validate_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolValidationResult:
    """Allow only known tools with strictly validated, normalized arguments."""

    argument_model = _TOOL_ARGUMENT_MODELS.get(tool_name)
    if argument_model is None:
        return ToolValidationResult(
            valid=False,
            tool_name=tool_name,
            error=f"Tool {tool_name!r} is not allowlisted.",
        )

    try:
        validated = argument_model.model_validate(arguments)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors()
        )
        return ToolValidationResult(
            valid=False,
            tool_name=tool_name,
            error=f"Invalid tool arguments: {details}",
        )

    return ToolValidationResult(
        valid=True,
        tool_name=tool_name,
        normalized_arguments=validated.model_dump(exclude_none=True),
    )
