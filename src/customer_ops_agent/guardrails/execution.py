"""Execution controls for bounded, recoverable tool orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from threading import Lock
from typing import Any

from customer_ops_agent.observability import record_circuit_state

from .models import (
    CircuitBreakerResult,
    CircuitState,
    DeadLoopResult,
    RetryResult,
    ToolCallLimitResult,
)


class ToolCallLimiter:
    """Enforce a hard per-request tool-call budget."""

    def __init__(self, maximum_calls: int = 8) -> None:
        if maximum_calls < 1:
            raise ValueError("maximum_calls must be at least 1")
        self.maximum_calls = maximum_calls
        self._calls_used = 0
        self._lock = Lock()

    def record_call(self) -> ToolCallLimitResult:
        """Reserve one call before execution, or reject when exhausted."""

        with self._lock:
            if self._calls_used >= self.maximum_calls:
                return ToolCallLimitResult(
                    allowed=False,
                    calls_used=self._calls_used,
                    calls_remaining=0,
                    error=f"Maximum of {self.maximum_calls} tool calls reached.",
                )
            self._calls_used += 1
            return ToolCallLimitResult(
                allowed=True,
                calls_used=self._calls_used,
                calls_remaining=self.maximum_calls - self._calls_used,
            )


class DeadLoopDetector:
    """Detect repeated node/tool/argument combinations in a rolling window."""

    def __init__(
        self,
        maximum_occurrences: int = 3,
        window_size: int = 10,
    ) -> None:
        if maximum_occurrences < 2:
            raise ValueError("maximum_occurrences must be at least 2")
        if window_size < maximum_occurrences:
            raise ValueError("window_size must cover maximum_occurrences")
        self.maximum_occurrences = maximum_occurrences
        self._history: deque[str] = deque(maxlen=window_size)
        self._lock = Lock()

    def record(
        self,
        node: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> DeadLoopResult:
        """Record an action and flag it when repetition reaches the limit."""

        canonical = json.dumps(
            {"node": node, "tool": tool_name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        # A digest is sufficient for equality checks and avoids retaining or
        # returning sensitive argument values in guardrail state.
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        with self._lock:
            self._history.append(signature)
            occurrences = self._history.count(signature)

        detected = occurrences >= self.maximum_occurrences
        return DeadLoopResult(
            dead_loop_detected=detected,
            signature=signature,
            occurrences=occurrences,
            error="Repeated action loop detected." if detected else None,
        )


def execute_with_retry(
    operation: Callable[[], Any],
    *,
    maximum_attempts: int = 3,
    base_delay_seconds: float = 0.25,
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    sleeper: Callable[[float], None] = time.sleep,
) -> RetryResult:
    """Retry only declared transient failures with bounded exponential backoff."""

    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be at least 1")
    if base_delay_seconds < 0:
        raise ValueError("base_delay_seconds must not be negative")

    for attempt in range(1, maximum_attempts + 1):
        try:
            return RetryResult(success=True, attempts=attempt, value=operation())
        except retry_on as exc:
            if attempt == maximum_attempts:
                return RetryResult(
                    success=False,
                    attempts=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
            sleeper(base_delay_seconds * (2 ** (attempt - 1)))
        except Exception as exc:
            # Business and validation errors are not retried because repetition
            # cannot make an invalid request safe or correct.
            return RetryResult(
                success=False,
                attempts=attempt,
                error=f"{type(exc).__name__}: {exc}",
            )

    raise AssertionError("Retry loop exited unexpectedly.")


class CircuitBreaker:
    """Stop calls to an unhealthy dependency and probe after a cooldown."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
        name: str = "default",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout_seconds < 0:
            raise ValueError("recovery_timeout_seconds must not be negative")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.name = name
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_probe_running = False
        self._lock = Lock()
        record_circuit_state(self.name, self._state.value)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def call(self, operation: Callable[[], Any]) -> CircuitBreakerResult:
        """Execute when permitted and update circuit state from the outcome."""

        with self._lock:
            now = self._clock()
            if self._state == CircuitState.OPEN:
                elapsed = now - self._opened_at
                if elapsed < self.recovery_timeout_seconds:
                    return CircuitBreakerResult(
                        success=False,
                        state=self._state,
                        rejected=True,
                        error="Circuit is open; dependency call was not attempted.",
                    )
                self._state = CircuitState.HALF_OPEN
                record_circuit_state(self.name, self._state.value)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_running:
                    return CircuitBreakerResult(
                        success=False,
                        state=self._state,
                        rejected=True,
                        error="Half-open circuit already has a probe in progress.",
                    )
                self._half_open_probe_running = True

        try:
            value = operation()
        except Exception as exc:
            with self._lock:
                self._failure_count += 1
                self._half_open_probe_running = False
                if (
                    self._state == CircuitState.HALF_OPEN
                    or self._failure_count >= self.failure_threshold
                ):
                    self._state = CircuitState.OPEN
                    self._opened_at = self._clock()
                state = self._state
                record_circuit_state(self.name, state.value)
            return CircuitBreakerResult(
                success=False,
                state=state,
                error=f"{type(exc).__name__}: {exc}",
            )

        with self._lock:
            self._failure_count = 0
            self._half_open_probe_running = False
            self._state = CircuitState.CLOSED
            record_circuit_state(self.name, self._state.value)
        return CircuitBreakerResult(
            success=True,
            state=CircuitState.CLOSED,
            value=value,
        )
