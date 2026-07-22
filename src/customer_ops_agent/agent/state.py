"""Shared state contracts for the Customer Operations Agent graph."""

from __future__ import annotations

from typing import Required, TypedDict

from customer_ops_agent.guardrails.models import (
    AgentOutput,
    InputGuardrailResult,
    OutputValidationResult,
)
from customer_ops_agent.memory.models import MemoryContext

from .models import HandlerResult, Route


class AgentState(TypedDict, total=False):
    """Minimal state passed between LangGraph nodes.

    Only the inbound fields are required. Each node owns the fields it adds,
    which keeps future handler changes isolated.
    """

    message: Required[str]
    conversation_id: Required[str]
    customer_id: str
    memory_context: MemoryContext
    sanitized_message: str
    input_guardrail: InputGuardrailResult
    route: Route
    route_reason: str
    planner_error: str
    handler_result: HandlerResult
    output_candidate: AgentOutput
    output_validation: OutputValidationResult
    response: str
