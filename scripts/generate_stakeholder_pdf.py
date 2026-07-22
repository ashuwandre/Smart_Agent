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
SAFE = HexColor("#177245")


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
        "Tier 1 + Tier 2 architecture, controls, and engineering tradeoffs",
    )

    y = heading(canvas, "Architecture", PAGE_HEIGHT - 98)
    available = PAGE_WIDTH - 2 * MARGIN
    gap = 10
    box_width = (available - 4 * gap) / 5
    row_one = [
        "Customer input",
        "Input guardrails",
        "Memory context",
        "LLM planner",
        "Specialist handler",
    ]
    row_two = [
        "LLM tool loop",
        "Tool validator",
        "Business rules",
        "Mock tools / FAISS",
        "Output validator",
    ]
    x_positions = [
        MARGIN + index * (box_width + gap) for index in range(5)
    ]
    top_y = y - 45
    lower_y = top_y - 66
    for index, label in enumerate(row_one):
        box(canvas, x_positions[index], top_y, box_width, label)
        if index < 4:
            arrow(
                canvas,
                x_positions[index] + box_width,
                top_y + 19,
                x_positions[index + 1],
            )
    for index, label in enumerate(row_two):
        box(canvas, x_positions[index], lower_y, box_width, label)
        if index < 4:
            arrow(
                canvas,
                x_positions[index] + box_width,
                lower_y + 19,
                x_positions[index + 1],
            )
    canvas.setStrokeColor(ACCENT)
    canvas.line(
        x_positions[-1] + box_width / 2,
        top_y,
        x_positions[-1] + box_width / 2,
        lower_y + 38,
    )
    canvas.line(
        x_positions[-1] + box_width / 2,
        lower_y + 38,
        x_positions[0] + box_width / 2,
        lower_y + 38,
    )
    canvas.line(
        x_positions[0] + box_width / 2,
        lower_y + 38,
        x_positions[0] + box_width / 2,
        lower_y + 34,
    )

    bar_y = lower_y - 34
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
            "One graph-based agent with an LLM router and focused specialists: explicit control flow and simpler recovery, at the cost of two model stages.",
            "The model chooses tools, but deterministic Pydantic validation and the BusinessRuleEngine decide whether actions execute.",
            "FAISS and JSON keep the assignment self-contained and inexpensive; they are not multi-process or horizontally scalable stores.",
            "Mock business systems make runs safe and repeatable while preserving realistic budgets, approval gates, tickets, and audit traces.",
            "Policy answers fail closed without KB citations; this reduces hallucination risk but can increase safe escalations when retrieval fails.",
        ],
        y,
    )

    y = heading(canvas, "Guardrail approach", y - 2)
    bullets(
        canvas,
        [
            "Before the LLM: prompt-injection detection, PII masking, abuse review, and blocked-input routing.",
            "Before each action: tool allowlist, strict arguments, customer isolation, maximum calls, dead-loop detection, and circuit breaking.",
            "Financial controls: refunds above INR 1,000 require human handling; offers are at most 20% with a total INR 10,000 run budget.",
            "Before delivery: structured output schema, confidence threshold, citation requirement, PII masking, and safe fallback.",
        ],
        y,
    )
    footer(canvas, 1)


def draw_page_two(canvas: Canvas) -> None:
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 48, "Evidence, rollout, cost, and risk")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        MARGIN,
        PAGE_HEIGHT - 65,
        "Submission scope: Tier 1 and Tier 2; Tier 3 is not claimed",
    )

    y = heading(canvas, "Verification evidence", PAGE_HEIGHT - 96)
    y = bullets(
        canvas,
        [
            "42 deterministic tests pass, including 12 integration scenarios for billing, refund, cancellation, technical support, injection, abuse, ambiguity, budgets, approval, circuit breaking, and loops.",
            "Scripted model tests prove genuine tool-selection flow: customer lookup, KB search, grounded citation capture, cross-customer rejection, and large-refund escalation.",
            "Offline CLI and Streamlit dashboard render successfully; every graph request and tool call is logged.",
            "Live OpenAI connectivity reached the provider but returned 429 insufficient_quota. The graph logged the failure and returned a safe human-support response.",
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
            "PII and payment data create privacy risk; mask before model use, minimize logs, encrypt durable stores, and apply retention policies.",
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

    canvas.setFillColor(SAFE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(
        MARGIN,
        55,
        "Current recommendation: demonstrate with a funded API key, then record the required 2-4 minute live video.",
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
