"""Pluggable planner implementations for graph routing."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from .state import Route


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class RouteDecision(BaseModel):
    """Structured planner output prevents free-form route parsing."""

    route: Route
    reason: str = Field(min_length=1, max_length=240)


Planner = Callable[[str], RouteDecision]


class LLMPlanner:
    """Use an OpenAI model only to select the appropriate specialist handler."""

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        # The model remains configurable without widening graph state or coupling
        # orchestration to deployment configuration.
        self.model = model or os.getenv("PLANNER_MODEL", "gpt-4o-mini")
        self._client = client

    def __call__(self, message: str) -> RouteDecision:
        client = self._client or OpenAI()
        response = client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Route a subscription customer request to exactly one "
                        "handler. Use billing for invoices, charges, plan changes, "
                        "and account questions; refund for requests to return "
                        "money; technical for connectivity, device, password, or "
                        "playback problems; retention for cancellation or churn "
                        "threats. Return a brief routing reason."
                    ),
                },
                {"role": "user", "content": message},
            ],
            text_format=RouteDecision,
        )
        if response.output_parsed is None:
            raise RuntimeError("Planner model returned no structured route.")
        return response.output_parsed


class MockPlanner:
    """Provide deterministic offline routing for graph tests and local demos."""

    def __call__(self, message: str) -> RouteDecision:
        # Graph memory is included for parity with the LLM planner, but this
        # keyword mock routes on the current turn to avoid stale-topic matches.
        current_message = message.rsplit("Current customer message:\n", 1)[-1]
        normalized = current_message.lower()
        if any(term in normalized for term in ("refund", "money back", "reimburse")):
            route: Route = "refund"
        elif any(term in normalized for term in ("cancel", "leave", "too expensive")):
            route = "retention"
        elif any(
            term in normalized
            for term in (
                "wifi",
                "wi-fi",
                "router",
                "speed",
                "network",
                "password",
                "buffer",
                "error",
            )
        ):
            route = "technical"
        else:
            route = "billing"

        return RouteDecision(
            route=route,
            reason=f"Mock planner matched the request to the {route} handler.",
        )
