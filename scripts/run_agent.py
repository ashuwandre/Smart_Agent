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
    HumanApprovalRequest,
    LLMToolCallingHandler,
    MockPlanner,
    MockSpecialistHandler,
    build_graph,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Customer Operations Agent")
    parser.add_argument("--customer-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument(
        "--conversation-id",
        default=None,
        help=(
            "Stable ID for multi-turn memory. Defaults to a customer-specific "
            "CLI conversation."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic doubles; no model or tool actions are executed.",
    )
    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Prompt a human for yes/no approval when a large refund is requested.",
    )
    return parser.parse_args()


def request_human_approval(request: HumanApprovalRequest) -> bool:
    """Collect an explicit decision outside model context."""

    print("\nHUMAN APPROVAL REQUIRED")
    print(f"Customer: {request.customer_id}")
    print(f"Refund amount: {request.amount:.2f}")
    print(f"Reason: {request.reason}")
    while True:
        decision = input("Approve this refund? [y/N]: ").strip().lower()
        if decision in ("y", "yes"):
            return True
        if decision in ("", "n", "no"):
            return False
        print("Enter yes or no.")


def main() -> None:
    arguments = parse_arguments()
    graph_options = {}
    if arguments.offline:
        graph_options = {
            "planner": MockPlanner(),
            "specialist_handler": MockSpecialistHandler(),
        }
    elif arguments.interactive_approval:
        graph_options = {
            "specialist_handler": LLMToolCallingHandler(
                approval_callback=request_human_approval
            )
        }
    graph = build_graph(**graph_options)
    conversation_id = (
        arguments.conversation_id
        or f"cli-{arguments.customer_id.strip().lower()}"
    )
    result = graph.invoke(
        {
            "conversation_id": conversation_id,
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
