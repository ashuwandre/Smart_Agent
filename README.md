# Customer Operations Agent

A safe, observable customer-support agent prototype for a subscription business. It routes billing, refund, technical, and retention requests through LangGraph and includes local mock tools, RAG, guardrails, memory, business rules, tests, and a Streamlit dashboard.

## Technology choices

- **LangGraph:** explicit routing and state transitions are easier to inspect and test than a single large prompt.
- **OpenAI:** `gpt-4o-mini` performs structured planner routing. The model can be changed with `PLANNER_MODEL`.
- **OpenAI Embeddings:** `text-embedding-3-small` provides cost-effective semantic search for the policy knowledge base.
- **FAISS:** fast local vector search without an external database.
- **Pydantic:** validates planner decisions, tool results, actions, guardrails, and logs.
- **Streamlit:** provides a lightweight local operations dashboard.

## Agent design

- The goal is to resolve a request with a grounded answer or confirmed action; otherwise the request ends in clarification or human escalation.
- One Customer Operations Agent uses an LLM planner and four focused handlers, avoiding unnecessary independent-agent coordination.
- Specialist LLMs select from route-specific tools; deterministic code validates and executes every selected action.
- Customer and policy facts come only from mock customer data and KB search results.
- Refunds above the threshold, unsafe inputs, low-confidence outputs, and provider failures remain human-controlled.
- A request is done only after output validation, conversation-memory persistence, and observability logging.

## Project structure

```text
data/
  knowledge_base/       Policy and troubleshooting documents
  synthetic/            Customers, inbox messages, and demo scenarios
src/customer_ops_agent/
  agent/                LangGraph planner and routing
  business_rules/       Financial and eligibility enforcement
  guardrails/           Input, output, tool, and execution controls
  memory/               JSON conversation memory
  observability/        CSV and JSON telemetry
  rag/                  Markdown chunking, embeddings, and FAISS search
  tools/                Deterministic mock customer-operation tools
dashboard/              Streamlit dashboard
tests/                  Unit and integration scenarios
```

## Submission documents

- `docs/RUNBOOK.md` — complete setup and agent/dashboard run instructions.
- `docs/TEST_SCENARIOS.md` — manual and automated test scenarios.
- `docs/stakeholder_summary.pdf` — two-page architecture and stakeholder summary.

Regenerate the PDF after changing its source script:

```powershell
python scripts/generate_stakeholder_pdf.py
```

## Setup

Requirements: Python 3.14 or a compatible Python 3 version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add your key:

```powershell
Copy-Item .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key
PLANNER_MODEL=gpt-4o-mini
AGENT_MODEL=gpt-4o-mini
```

The model settings are optional. The OpenAI account must have available API
quota. Never commit the `.env` file.

## Run the graph

Set the source directory on `PYTHONPATH`, then open Python:

```powershell
$env:PYTHONPATH="src"
python
```

```python
from customer_ops_agent.agent import build_graph

graph = build_graph()
result = graph.invoke(
    {
        "conversation_id": "demo-001",
        "customer_id": "CUST0003",
        "message": "Why is my bill higher this month?",
    }
)
print(result["response"])
```

The default planner and specialist handlers make OpenAI API requests. The
specialist model chooses tools, while guardrails and the business-rule engine
decide whether each action may execute.

For a concise JSON execution trace:

```powershell
python scripts/run_agent.py --customer-id CUST0003 --message "Why is my bill higher this month?"
```

For an API-free structure check that executes no customer action:

```powershell
python scripts/run_agent.py --offline --customer-id CUST0003 --message "Why is my bill higher this month?"
```

## Run tests

```powershell
python -m pytest
```

The suite covers billing, refunds, cancellation, technical issues, prompt injection, abusive and ambiguous requests, budget exhaustion, human approval, circuit breaking, and dead-loop detection.

## Run the dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard displays tool timelines, remaining retention budget, circuit-breaker state, pending human approvals, latency, and errors.

## Business controls

- Refunds above **₹1,000** require trusted human approval.
- Retention offers cannot exceed **20%**.
- Total retention-offer cost cannot exceed **₹10,000 per run**.
- Retention eligibility requires at least **12 months tenure**, churn risk of **0.70**, an active account, and no outstanding invoice.
- Prompt injection is blocked before planner execution.
- PII is masked before model processing and customer-facing output.
- Tool calls are allowlisted, bounded, retry-controlled, and loop-checked.

## Tier status

- **Tier 1 — Implemented:** the LLM planner routes requests, specialist LLMs select tools, tool results are observable, and policy responses require KB citations.
- **Tier 2 — Implemented:** explicit graph orchestration, bounded multi-turn JSON memory, retention controls, refund approval routing, refusals, injection resistance, and structured validated outputs are included.
- **Tier 3 — Not claimed:** adversarial tests and observability are useful engineering additions, but no formal evaluation scorecard or productionization deliverable is claimed.

Automated tests use scripted model decisions so they are deterministic and do
not consume API quota. A funded OpenAI key is required for a live LLM run.

## Assumptions and limitations

- Customer systems, tickets, refunds, and offers are deterministic local mocks.
- Conversation memory uses one local JSON file and is intended for a single-process demo.
- Human approval is represented as a structured pending state; there is no external approval service or UI.
- Specialist confidence is model-reported and is not statistically calibrated.
- FAISS is built in memory and must be rebuilt when the process restarts.
- Local CSV/JSON observability is suitable for the assignment, not a distributed production deployment.
- Live verification depends on OpenAI account quota; provider failures return a safe human-support fallback.
