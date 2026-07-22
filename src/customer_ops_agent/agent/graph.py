"""LangGraph orchestration for one routed Customer Operations Agent."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from customer_ops_agent.business_rules import BusinessRuleEngine
from customer_ops_agent.guardrails import (
    AgentOutput,
    GuardrailDecision,
    evaluate_input,
    validate_output,
)
from customer_ops_agent.memory import ConversationMemoryStore
from customer_ops_agent.observability import (
    DEFAULT_OBSERVABILITY_STORE,
    ObservableGraph,
    ObservabilityStore,
)

from .handlers import LLMToolCallingHandler, SpecialistHandler
from .models import HandlerResult, Route
from .planner import LLMPlanner, Planner
from .state import AgentState


LOAD_MEMORY_NODE = "load_memory"
INPUT_GUARDRAIL_NODE = "input_guardrail"
PLANNER_NODE = "planner"
PLANNER_FAILURE_NODE = "planner_failure"
BILLING_NODE = "billing_handler"
REFUND_NODE = "refund_handler"
TECHNICAL_NODE = "technical_handler"
RETENTION_NODE = "retention_handler"
BLOCKED_RESPONSE_NODE = "blocked_response"
RESPONSE_NODE = "response_generator"
OUTPUT_GUARDRAIL_NODE = "output_guardrail"
SAVE_MEMORY_NODE = "save_memory"


def _load_memory(
    memory_store: ConversationMemoryStore,
) -> Callable[[AgentState], dict[str, object]]:
    """Load prior safe interactions and restore a known customer identifier."""

    def load(state: AgentState) -> dict[str, object]:
        context = memory_store.get_context(state["conversation_id"])
        current_customer = state.get("customer_id")
        if (
            current_customer
            and context.customer_id
            and current_customer.strip().upper() != context.customer_id
        ):
            raise ValueError("Conversation customer_id does not match stored memory.")

        updates: dict[str, object] = {"memory_context": context}
        if not current_customer and context.customer_id:
            updates["customer_id"] = context.customer_id
        return updates

    return load


def _input_guardrail(state: AgentState) -> dict[str, object]:
    """Screen and mask inbound text before any model receives it."""

    result = evaluate_input(state["message"])
    return {
        "input_guardrail": result,
        "sanitized_message": result.sanitized_text,
    }


def _input_decision(state: AgentState) -> str:
    """Route blocked inputs away from the planner and all specialist nodes."""

    return (
        "blocked"
        if state["input_guardrail"].decision == GuardrailDecision.BLOCK
        else "continue"
    )


def _planner_node(planner: Planner) -> Callable[[AgentState], dict[str, object]]:
    """Adapt the injected planner to the LangGraph state-update contract."""

    def plan(state: AgentState) -> dict[str, object]:
        message = state["sanitized_message"].strip()
        if not message:
            raise ValueError("message must not be empty")

        previous_messages = state["memory_context"].previous_messages
        history = "\n".join(
            f"{interaction.role}: {interaction.message}"
            for interaction in previous_messages
        )
        planner_input = (
            f"Previous conversation:\n{history}\n\n"
            f"Current customer message:\n{message}"
            if history
            else f"Current customer message:\n{message}"
        )
        try:
            decision = planner(planner_input)
        except Exception as exc:
            return {
                "planner_error": (
                    f"Planner unavailable: {type(exc).__name__}."
                )
            }
        return {"route": decision.route, "route_reason": decision.reason}

    return plan


def _specialist_node(
    route: Route,
    specialist_handler: SpecialistHandler,
) -> Callable[[AgentState], dict[str, HandlerResult]]:
    """Adapt one shared LLM tool runner to a route-specific graph node."""

    def handle(state: AgentState) -> dict[str, HandlerResult]:
        result = specialist_handler(route, state)
        return {"handler_result": result}

    return handle


def _blocked_response(state: AgentState) -> dict[str, AgentOutput]:
    """Create a safe terminal candidate without exposing detector details."""

    candidate = AgentOutput(
        response=(
            "I cannot process those instructions automatically. "
            "The request must be handled safely by human support."
        ),
        action="refuse",
        confidence=1,
    )
    return {"output_candidate": candidate}


def _planner_failure_response(state: AgentState) -> dict[str, AgentOutput]:
    """Fail safely when the routing model or provider is unavailable."""

    return {
        "output_candidate": AgentOutput(
            response=(
                "I cannot process this request automatically right now. "
                "Please route it to human support."
            ),
            action="escalate",
            confidence=1,
        )
    }


def _response_generator(state: AgentState) -> dict[str, AgentOutput]:
    """Turn the specialist result into a grounded structured output candidate."""

    result = state["handler_result"]
    response = result.summary
    if result.citations:
        response += "\n\nSources:\n" + "\n".join(
            f"- {citation}" for citation in result.citations
        )
    return {
        "output_candidate": AgentOutput(
            response=response,
            action=result.action,
            confidence=result.confidence,
            citations=result.citations,
            policy_based=result.policy_based,
        )
    }


def _output_guardrail(state: AgentState) -> dict[str, object]:
    """Validate the structured candidate and expose only a safe response."""

    validation = validate_output(state["output_candidate"])
    if validation.valid and validation.output is not None:
        response = validation.output.response
    else:
        response = (
            "I cannot provide a reliable response automatically. "
            "The request requires human support."
        )
    return {"output_validation": validation, "response": response}


def _save_memory(
    memory_store: ConversationMemoryStore,
) -> Callable[[AgentState], dict[str, object]]:
    """Atomically save the customer and assistant sides of the completed turn."""

    def save(state: AgentState) -> dict[str, object]:
        input_allowed = (
            state["input_guardrail"].decision != GuardrailDecision.BLOCK
        )
        memory_store.append_turn(
            state["conversation_id"],
            state["sanitized_message"],
            state["response"],
            customer_id=state.get("customer_id"),
            # Blocked hostile instructions are retained for audit continuity but
            # never fed back into a future planner prompt.
            customer_include_in_context=input_allowed,
        )
        handler_result = state.get("handler_result")
        if handler_result is not None:
            for trace in handler_result.tool_calls:
                if (
                    trace.tool == "grant_retention_offer"
                    and trace.executed
                    and "pct" in trace.arguments
                ):
                    memory_store.record_offer(
                        state["conversation_id"],
                        float(trace.arguments["pct"]),
                        trace.status,
                        customer_id=state.get("customer_id"),
                    )
                elif (
                    trace.tool == "issue_refund"
                    and "amount" in trace.arguments
                ):
                    memory_store.record_refund(
                        state["conversation_id"],
                        float(trace.arguments["amount"]),
                        trace.status,
                        customer_id=state.get("customer_id"),
                    )
            for step in handler_result.technical_steps:
                memory_store.record_technical_step(
                    state["conversation_id"],
                    step,
                    customer_id=state.get("customer_id"),
                )
        return {"memory_context": memory_store.get_context(state["conversation_id"])}

    return save


def _planner_decision(state: AgentState) -> str:
    """Route a valid plan or divert provider failures to a safe response."""

    return "failed" if state.get("planner_error") else state["route"]


def build_graph(
    planner: Planner | None = None,
    specialist_handler: SpecialistHandler | None = None,
    memory_store: ConversationMemoryStore | None = None,
    observability_store: ObservabilityStore | None = None,
):
    """Compile the single-agent planner-to-handler orchestration graph.

    The default planner and specialist are LLM-backed. Tests can inject offline
    doubles without changing graph topology or production controls.
    """

    selected_planner = planner or LLMPlanner()
    selected_memory_store = memory_store or ConversationMemoryStore()
    selected_observability_store = (
        observability_store or DEFAULT_OBSERVABILITY_STORE
    )
    selected_specialist_handler = specialist_handler or LLMToolCallingHandler(
        rule_engine=BusinessRuleEngine(
            observability_store=selected_observability_store
        )
    )
    graph = StateGraph(AgentState)

    graph.add_node(LOAD_MEMORY_NODE, _load_memory(selected_memory_store))
    graph.add_node(INPUT_GUARDRAIL_NODE, _input_guardrail)
    graph.add_node(PLANNER_NODE, _planner_node(selected_planner))
    graph.add_node(PLANNER_FAILURE_NODE, _planner_failure_response)
    graph.add_node(
        BILLING_NODE,
        _specialist_node("billing", selected_specialist_handler),
    )
    graph.add_node(
        REFUND_NODE,
        _specialist_node("refund", selected_specialist_handler),
    )
    graph.add_node(
        TECHNICAL_NODE,
        _specialist_node("technical", selected_specialist_handler),
    )
    graph.add_node(
        RETENTION_NODE,
        _specialist_node("retention", selected_specialist_handler),
    )
    graph.add_node(BLOCKED_RESPONSE_NODE, _blocked_response)
    graph.add_node(RESPONSE_NODE, _response_generator)
    graph.add_node(OUTPUT_GUARDRAIL_NODE, _output_guardrail)
    graph.add_node(SAVE_MEMORY_NODE, _save_memory(selected_memory_store))

    graph.add_edge(START, LOAD_MEMORY_NODE)
    graph.add_edge(LOAD_MEMORY_NODE, INPUT_GUARDRAIL_NODE)
    graph.add_conditional_edges(
        INPUT_GUARDRAIL_NODE,
        _input_decision,
        {
            "blocked": BLOCKED_RESPONSE_NODE,
            "continue": PLANNER_NODE,
        },
    )
    graph.add_conditional_edges(
        PLANNER_NODE,
        _planner_decision,
        {
            "billing": BILLING_NODE,
            "refund": REFUND_NODE,
            "technical": TECHNICAL_NODE,
            "retention": RETENTION_NODE,
            "failed": PLANNER_FAILURE_NODE,
        },
    )

    # Every specialist rejoins one response node, keeping response formatting
    # consistent without duplicating terminal behavior.
    for handler_node in (
        BILLING_NODE,
        REFUND_NODE,
        TECHNICAL_NODE,
        RETENTION_NODE,
    ):
        graph.add_edge(handler_node, RESPONSE_NODE)
    graph.add_edge(BLOCKED_RESPONSE_NODE, OUTPUT_GUARDRAIL_NODE)
    graph.add_edge(PLANNER_FAILURE_NODE, OUTPUT_GUARDRAIL_NODE)
    graph.add_edge(RESPONSE_NODE, OUTPUT_GUARDRAIL_NODE)
    graph.add_edge(OUTPUT_GUARDRAIL_NODE, SAVE_MEMORY_NODE)
    graph.add_edge(SAVE_MEMORY_NODE, END)

    compiled_graph = graph.compile()
    return ObservableGraph(
        compiled_graph,
        selected_observability_store,
        sanitize=lambda text: evaluate_input(text).sanitized_text,
    )
