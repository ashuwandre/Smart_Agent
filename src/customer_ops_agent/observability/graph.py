"""Observable wrapper that logs successful, blocked, and failed graph requests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from .models import ConversationLog
from .store import ObservabilityStore, circuit_state_snapshot


class ObservableGraph:
    """Delegate to a compiled graph while guaranteeing request-level telemetry."""

    def __init__(
        self,
        graph: Any,
        store: ObservabilityStore,
        sanitize: Callable[[str], str],
    ) -> None:
        self._graph = graph
        self._store = store
        self._sanitize = sanitize

    def _log(
        self,
        request: dict[str, Any],
        result: dict[str, Any] | None,
        started_at: float,
        error: Exception | None,
    ) -> None:
        guardrail = result.get("input_guardrail") if result else None
        decision = str(guardrail.decision) if guardrail else None
        candidate = result.get("output_candidate") if result else None
        planner_error = result.get("planner_error") if result else None
        status = (
            "error"
            if error or planner_error
            else "blocked"
            if decision == "block"
            else "completed"
        )
        self._store.log_conversation(
            ConversationLog(
                timestamp=datetime.now(UTC),
                conversation_id=str(request.get("conversation_id", "unknown")),
                customer=request.get("customer_id"),
                request=self._sanitize(str(request.get("message", ""))),
                response=result.get("response") if result else None,
                route=result.get("route") if result else None,
                status=status,
                confidence=(
                    float(candidate.confidence)
                    if candidate is not None
                    else None
                ),
                latency=round((perf_counter() - started_at) * 1_000, 3),
                guardrail_decision=decision,
                error=(
                    f"{type(error).__name__}: {error}"
                    if error
                    else planner_error
                ),
                circuit_breakers=circuit_state_snapshot(),
            )
        )

    def invoke(
        self,
        input: dict[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Log one synchronous request even when graph execution raises."""

        started_at = perf_counter()
        try:
            result = self._graph.invoke(input, config=config, **kwargs)
        except Exception as exc:
            self._log(input, None, started_at, exc)
            raise
        self._log(input, result, started_at, None)
        return result

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Log one asynchronous request even when graph execution raises."""

        started_at = perf_counter()
        try:
            result = await self._graph.ainvoke(input, config=config, **kwargs)
        except Exception as exc:
            self._log(input, None, started_at, exc)
            raise
        self._log(input, result, started_at, None)
        return result

    def __getattr__(self, name: str) -> Any:
        """Preserve access to visualization and other compiled-graph APIs."""

        return getattr(self._graph, name)
