"""Structured JSONL audit logging shared by every mock tool."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import ParamSpec, TypeVar

from pydantic import BaseModel

from customer_ops_agent.observability import (
    DEFAULT_OBSERVABILITY_STORE,
    ObservabilityStore,
    ToolCallEvent,
)


P = ParamSpec("P")
R = TypeVar("R", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_LOG_PATH = PROJECT_ROOT / "artifacts" / "logs" / "mock_tools.jsonl"
TOOL_OBSERVABILITY_STORE: ObservabilityStore = DEFAULT_OBSERVABILITY_STORE
_log_lock = Lock()


@dataclass(frozen=True)
class ToolTelemetryContext:
    """Per-request values not present in the tool's function arguments."""

    confidence: float | None = None
    retry: int = 0
    loop_count: int = 0


_telemetry_context: ContextVar[ToolTelemetryContext] = ContextVar(
    "tool_telemetry_context",
    default=ToolTelemetryContext(),
)


@contextmanager
def tool_telemetry_context(
    *,
    confidence: float | None = None,
    retry: int = 0,
    loop_count: int = 0,
):
    """Attach orchestration metadata to nested tool calls without changing APIs."""

    token = _telemetry_context.set(
        ToolTelemetryContext(
            confidence=confidence,
            retry=retry,
            loop_count=loop_count,
        )
    )
    try:
        yield
    finally:
        _telemetry_context.reset(token)


def _write_event(event: dict[str, object]) -> None:
    """Append one complete event under a lock to prevent interleaved JSON."""

    with _log_lock:
        TOOL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TOOL_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def audit_tool(function: Callable[P, R]) -> Callable[P, R]:
    """Log tool inputs, outputs, elapsed time, and expected or raised errors."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started_at = perf_counter()
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)

        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            timestamp = datetime.now(UTC)
            latency = round((perf_counter() - started_at) * 1_000, 3)
            error = f"{type(exc).__name__}: {exc}"
            _write_event(
                {
                    "timestamp": timestamp.isoformat(),
                    "tool": function.__name__,
                    "input": inputs,
                    "output": None,
                    "execution_time_ms": latency,
                    "error": error,
                }
            )
            context = _telemetry_context.get()
            TOOL_OBSERVABILITY_STORE.log_tool_call(
                ToolCallEvent(
                    timestamp=timestamp,
                    customer=inputs.get("customer_id"),
                    tool=function.__name__,
                    confidence=context.confidence,
                    latency=latency,
                    retry=context.retry,
                    status="error",
                    loop_count=context.loop_count,
                    reason=error,
                )
            )
            raise

        timestamp = datetime.now(UTC)
        latency = round((perf_counter() - started_at) * 1_000, 3)
        _write_event(
            {
                "timestamp": timestamp.isoformat(),
                "tool": function.__name__,
                "input": inputs,
                "output": result.model_dump(mode="json"),
                "execution_time_ms": latency,
                "error": result.error,
            }
        )
        context = _telemetry_context.get()
        TOOL_OBSERVABILITY_STORE.log_tool_call(
            ToolCallEvent(
                timestamp=timestamp,
                customer=inputs.get("customer_id"),
                tool=function.__name__,
                confidence=context.confidence,
                latency=latency,
                retry=context.retry,
                status=getattr(
                    result,
                    "status",
                    "success" if result.success else "error",
                ),
                loop_count=context.loop_count,
                reason=getattr(result, "reason", "") or result.error or "",
            )
        )
        return result

    return wrapper
