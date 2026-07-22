"""Run one observable customer request for local verification or recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# The repository uses a src layout but is intentionally not packaged for this
# short assignment, so the demo resolves imports relative to its own location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_ops_agent.agent import (  # noqa: E402
    MockPlanner,
    MockSpecialistHandler,
    build_graph,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Customer Operations Agent")
    parser.add_argument("--customer-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--conversation-id", default="cli-demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic doubles; no model or tool actions are executed.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    graph_options = (
        {
            "planner": MockPlanner(),
            "specialist_handler": MockSpecialistHandler(),
        }
        if arguments.offline
        else {}
    )
    graph = build_graph(**graph_options)
    result = graph.invoke(
        {
            "conversation_id": arguments.conversation_id,
            "customer_id": arguments.customer_id,
            "message": arguments.message,
        }
    )

    handler = result.get("handler_result")
    trace = {
        "route": result.get("route"),
        "routing_reason": result.get("route_reason"),
        "planner_error": result.get("planner_error"),
        "tool_calls": (
            [tool.model_dump(mode="json") for tool in handler.tool_calls]
            if handler
            else []
        ),
        "action": result["output_candidate"].action,
        "citations": result["output_candidate"].citations,
        "response": result["response"],
    }
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
