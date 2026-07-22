"""Generate the required two-page stakeholder summary PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "stakeholder_summary.pdf"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 40
TEXT = HexColor("#172033")
MUTED = HexColor("#556176")
ACCENT = HexColor("#1957A6")
LIGHT = HexColor("#EAF1FA")
LINE = HexColor("#AAB6C8")


def heading(canvas: Canvas, text: str, y: float, size: int = 14) -> float:
    canvas.setFillColor(ACCENT)
    canvas.setFont("Helvetica-Bold", size)
    canvas.drawString(MARGIN, y, text)
    return y - size - 8


def wrapped_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def bullets(
    canvas: Canvas,
    items: list[str],
    y: float,
    *,
    width: int = 92,
    size: int = 8,
    leading: int = 11,
) -> float:
    canvas.setFont("Helvetica", size)
    canvas.setFillColor(TEXT)
    for item in items:
        lines = wrapped_lines(item, width)
        canvas.setFillColor(ACCENT)
        canvas.drawString(MARGIN + 2, y, "-")
        canvas.setFillColor(TEXT)
        for line in lines:
            canvas.drawString(MARGIN + 13, y, line)
            y -= leading
        y -= 3
    return y


def box(canvas: Canvas, x: float, y: float, width: float, text: str) -> None:
    canvas.setFillColor(LIGHT)
    canvas.setStrokeColor(LINE)
    canvas.roundRect(x, y, width, 38, 5, fill=1, stroke=1)
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica-Bold", 7)
    lines = wrapped_lines(text, 16)
    baseline = y + 23 + (len(lines) - 1) * 4
    for line in lines[:3]:
        canvas.drawCentredString(x + width / 2, baseline, line)
        baseline -= 9


def arrow(canvas: Canvas, x1: float, y: float, x2: float) -> None:
    canvas.setStrokeColor(ACCENT)
    canvas.setFillColor(ACCENT)
    canvas.line(x1, y, x2, y)
    canvas.line(x2, y, x2 - 4, y + 3)
    canvas.line(x2, y, x2 - 4, y - 3)


def footer(canvas: Canvas, page_number: int) -> None:
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 30, PAGE_WIDTH - MARGIN, 30)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 18, "Customer Operations Agent - Stakeholder Summary")
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN,
        18,
        f"Page {page_number} of 2",
    )


def draw_page_one(canvas: Canvas) -> None:
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 48, "Customer Operations Agent")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(
        MARGIN,
        PAGE_HEIGHT - 66,
        "A simple view of architecture, controls, decisions, and tradeoffs",
    )

    y = heading(canvas, "Architecture", PAGE_HEIGHT - 98)
    available = PAGE_WIDTH - 2 * MARGIN
    main_gap = 5
    main_box_width = (available - 7 * main_gap) / 8
    main_flow = [
        "Customer input",
        "Load memory",
        "Input guardrails",
        "LLM planner",
        "Specialist route",
        "Response formatter",
        "Output guardrails",
        "Save memory",
    ]
    main_x_positions = [
        MARGIN + index * (main_box_width + main_gap) for index in range(8)
    ]
    main_y = y - 45
    for index, label in enumerate(main_flow):
        box(canvas, main_x_positions[index], main_y, main_box_width, label)
        if index < 7:
            arrow(
                canvas,
                main_x_positions[index] + main_box_width,
                main_y + 19,
                main_x_positions[index + 1],
            )

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawString(MARGIN, main_y - 14, "INSIDE THE SPECIALIST'S BOUNDED TOOL LOOP")
    tool_gap = 10
    tool_box_width = (available - 3 * tool_gap) / 4
    tool_flow = [
        "LLM tool choice",
        "Tool guardrails",
        "Business rules",
        "Mock systems / RAG",
    ]
    tool_x_positions = [
        MARGIN + index * (tool_box_width + tool_gap) for index in range(4)
    ]
    tool_y = main_y - 65
    for index, label in enumerate(tool_flow):
        box(canvas, tool_x_positions[index], tool_y, tool_box_width, label)
        if index < 3:
            arrow(
                canvas,
                tool_x_positions[index] + tool_box_width,
                tool_y + 19,
                tool_x_positions[index + 1],
            )

    bar_y = tool_y - 34
    canvas.setFillColor(HexColor("#F4F6F9"))
    canvas.setStrokeColor(LINE)
    canvas.roundRect(MARGIN, bar_y, available, 24, 4, fill=1, stroke=1)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawCentredString(
        PAGE_WIDTH / 2,
        bar_y + 8,
        "JSON memory + CSV/JSON observability + Streamlit dashboard",
    )

    y = heading(canvas, "Key design decisions and tradeoffs", bar_y - 27)
    y = bullets(
        canvas,
        [
            "LangGraph was chosen instead of one large prompt: each step is visible, testable, and recoverable; the tradeoff is more orchestration code and model calls.",
            "OpenAI performs intent routing and genuine tool selection; deterministic Pydantic validation and the BusinessRuleEngine retain authority over execution.",
            "FAISS provides fast local semantic retrieval and JSON keeps memory simple; both are inexpensive for a demo but not suitable as shared production stores.",
            "Mock CRM, billing, and ticket systems make actions safe and repeatable while preserving realistic budgets, approvals, IDs, and audit traces.",
            "Policy and troubleshooting answers fail closed without grounding; this reduces hallucination risk but may increase safe escalations when retrieval fails.",
        ],
        y,
    )

    y = heading(canvas, "Guardrail approach", y - 2)
    bullets(
        canvas,
        [
            "Before the LLM: prompt-injection detection, PII masking, abuse review, and blocked-input routing.",
            "Before each action: tool allowlist, strict arguments, customer isolation, maximum calls, dead-loop detection, and circuit breaking.",
            "Financial controls: refunds above INR 1,000 pause for trusted yes/no approval; offers are capped at 20% and share an INR 10,000 run budget.",
            "Before delivery: structured output schema, confidence threshold, citation requirement, PII masking, and safe fallback.",
        ],
        y,
    )
    footer(canvas, 1)


def draw_page_two(canvas: Canvas) -> None:
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 48, "Evaluation, rollout, cost, and risk")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        MARGIN,
        PAGE_HEIGHT - 65,
        "Submission scope: Tier 1 and Tier 2; Tier 3 is not claimed",
    )

    y = heading(canvas, "Evaluation results", PAGE_HEIGHT - 96)
    y = bullets(
        canvas,
        [
            "45 deterministic tests pass: 12 integration test cases across 11 scenario categories, plus unit coverage for routing, tools, memory, guardrails, RAG, observability, and HITL.",
            "Tool-loop tests prove model-selected customer lookup, KB search, grounded citation capture, strict customer isolation, and controlled action execution.",
            "HITL tests prove both paths: approval revalidates and issues the refund; rejection leaves it unexecuted and creates a human escalation.",
            "Failure scenarios verify prompt-injection blocking, budget exhaustion, circuit breaking, dead-loop detection, and safe provider fallback.",
        ],
        y,
    )

    y = heading(canvas, "Production rollout plan", y - 2)
    y = bullets(
        canvas,
        [
            "1. Replace local mocks with sandbox adapters while retaining the same validated tool contracts.",
            "2. Run in shadow mode: compare proposed actions with human decisions and block all writes.",
            "3. Canary low-risk billing and troubleshooting requests; keep refunds and offers approval-only.",
            "4. Add durable conversation, approval, and budget storage with idempotency keys and role-based access.",
            "5. Add distributed tracing, alerting, secrets management, model/version pinning, and retrieval-quality monitoring.",
            "6. Expand gradually only when safety, groundedness, latency, cost, and business KPI thresholds remain healthy.",
        ],
        y,
    )

    y = heading(canvas, "Cost and risk notes", y - 2)
    y = bullets(
        canvas,
        [
            "Primary variable costs are planner/specialist tokens and KB embeddings; cache stable embeddings and keep prompts/tool results compact.",
            "Retention discounts directly reduce revenue, so eligibility and aggregate budget are enforced in code rather than prompts.",
            "Provider quota, latency, and outages can interrupt automation; retries, circuit breakers, observability, and human fallback limit impact.",
            "PII and billing data create privacy risk; mask before model use, minimize logs, encrypt durable stores, and apply retention policies.",
            "Model confidence is self-reported and not calibrated; production thresholds require empirical validation against human-reviewed outcomes.",
        ],
        y,
    )

    y = heading(canvas, "Business KPIs", y - 2)
    y = bullets(
        canvas,
        [
            "Resolution rate and escalation rate by intent.",
            "Net retained monthly revenue minus discount cost.",
            "Cost per resolved request and p95 end-to-end latency.",
            "Grounding failures, approval violations, injection blocks, and incorrect tool calls.",
        ],
        y,
    )

    footer(canvas, 2)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUTPUT_PATH), pagesize=A4)
    canvas.setTitle("Customer Operations Agent - Stakeholder Summary")
    canvas.setAuthor("Customer Operations Agent Project")
    draw_page_one(canvas)
    canvas.showPage()
    draw_page_two(canvas)
    canvas.showPage()
    canvas.save()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
