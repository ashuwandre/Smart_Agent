"""Streamlit operations dashboard backed only by local telemetry files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"
TOOL_CALLS_PATH = LOG_DIR / "tool_calls.csv"
CONVERSATIONS_PATH = LOG_DIR / "conversation_logs.json"
TOOL_AUDIT_PATH = LOG_DIR / "mock_tools.jsonl"
RETENTION_RUN_BUDGET = 10_000.0


@st.cache_data(ttl=2)
def load_tool_calls() -> pd.DataFrame:
    if not TOOL_CALLS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TOOL_CALLS_PATH)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["latency"] = pd.to_numeric(frame["latency"], errors="coerce")
    return frame


@st.cache_data(ttl=2)
def load_conversations() -> list[dict[str, Any]]:
    if not CONVERSATIONS_PATH.exists():
        return []
    try:
        payload = json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload.get("requests", [])


@st.cache_data(ttl=2)
def load_tool_audit() -> list[dict[str, Any]]:
    if not TOOL_AUDIT_PATH.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in TOOL_AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # One malformed line must not make the operations dashboard unusable.
            continue
    return events


def budget_remaining(events: list[dict[str, Any]]) -> float:
    for event in reversed(events):
        if event.get("tool") != "grant_retention_offer":
            continue
        output = event.get("output") or {}
        if "remaining_budget" in output:
            return float(output["remaining_budget"])
    return RETENTION_RUN_BUDGET


def approval_queue(
    events: list[dict[str, Any]],
    tool_calls: pd.DataFrame,
) -> pd.DataFrame:
    # A later issued refund removes an earlier pending item for that customer.
    pending: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("tool") != "issue_refund":
            continue
        output = event.get("output") or {}
        customer_id = output.get("customer_id")
        if not customer_id:
            continue
        if output.get("status") == "pending_approval":
            pending[customer_id] = {
                "timestamp": event.get("timestamp"),
                "customer": customer_id,
                "amount": output.get("amount"),
                "reason": output.get("reason"),
            }
        elif output.get("status") == "issued":
            pending.pop(customer_id, None)

    if not tool_calls.empty:
        refund_rows = tool_calls[tool_calls["tool"] == "issue_refund"]
        for _, row in refund_rows.sort_values("timestamp").iterrows():
            customer_id = row.get("customer")
            if pd.isna(customer_id):
                continue
            if row.get("status") in (
                "requires_human_approval",
                "pending_approval",
            ):
                pending[str(customer_id)] = {
                    "timestamp": row.get("timestamp"),
                    "customer": customer_id,
                    "amount": None,
                    "reason": row.get("reason"),
                }
            elif row.get("status") == "issued":
                pending.pop(str(customer_id), None)
    return pd.DataFrame(pending.values())


st.set_page_config(page_title="Customer Operations", layout="wide")
st.title("Customer Operations Observability")
st.caption("Local tool and conversation telemetry")

if st.button("Refresh data"):
    st.cache_data.clear()

tool_calls = load_tool_calls()
conversations = load_conversations()
tool_audit = load_tool_audit()
queue = approval_queue(tool_audit, tool_calls)

latest_circuits = (
    conversations[-1].get("circuit_breakers", {}) if conversations else {}
)
overall_circuit_status = (
    "OPEN"
    if "open" in latest_circuits.values()
    else "HEALTHY"
    if latest_circuits
    else "NO DATA"
)
average_latency = (
    float(tool_calls["latency"].dropna().mean())
    if not tool_calls.empty and tool_calls["latency"].notna().any()
    else 0
)
error_count = (
    int(tool_calls["status"].isin(["error", "denied", "rejected"]).sum())
    if not tool_calls.empty
    else 0
)

budget_column, circuit_column, approval_column, latency_column, error_column = (
    st.columns(5)
)
budget_column.metric("Budget Remaining", f"₹{budget_remaining(tool_audit):,.2f}")
circuit_column.metric("Circuit Breakers", overall_circuit_status)
approval_column.metric("Human Approvals", len(queue))
latency_column.metric("Average Tool Latency", f"{average_latency:.2f} ms")
error_column.metric("Errors / Denials", error_count)

st.subheader("Tool Timeline")
if tool_calls.empty:
    st.info("No tool calls have been recorded yet.")
else:
    st.dataframe(
        tool_calls.sort_values("timestamp", ascending=False),
        width="stretch",
        hide_index=True,
    )

left, right = st.columns(2)
with left:
    st.subheader("Latency")
    if tool_calls.empty:
        st.info("Latency data will appear after the first tool call.")
    else:
        latency_by_tool = (
            tool_calls.dropna(subset=["latency"])
            .groupby("tool", as_index=False)["latency"]
            .mean()
            .set_index("tool")
        )
        st.bar_chart(latency_by_tool, y="latency")

with right:
    st.subheader("Circuit Breaker Status")
    if not latest_circuits:
        st.info("No circuit breaker state has been recorded.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {"circuit": name, "status": status}
                    for name, status in latest_circuits.items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )

st.subheader("Human Approval Queue")
if queue.empty:
    st.success("No refunds are waiting for human approval.")
else:
    st.dataframe(queue, width="stretch", hide_index=True)

st.subheader("Errors")
tool_errors = (
    tool_calls[tool_calls["status"].isin(["error", "denied", "rejected"])]
    if not tool_calls.empty
    else pd.DataFrame()
)
conversation_errors = [
    {
        "timestamp": item.get("timestamp"),
        "customer": item.get("customer"),
        "tool": "conversation",
        "status": item.get("status"),
        "reason": item.get("error"),
    }
    for item in conversations
    if item.get("status") == "error"
]
errors = pd.concat(
    [tool_errors, pd.DataFrame(conversation_errors)],
    ignore_index=True,
)
if errors.empty:
    st.success("No errors have been recorded.")
else:
    st.dataframe(errors, width="stretch", hide_index=True)
