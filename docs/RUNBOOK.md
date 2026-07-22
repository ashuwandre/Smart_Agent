# Runbook: Start the Application from Scratch

This project has two runnable surfaces:

1. **Customer Operations Agent CLI** — processes one customer request and prints a structured execution trace.
2. **Streamlit dashboard** — displays tool calls, approvals, latency, budget, circuit state, and errors.

## 1. Prerequisites

- Windows PowerShell
- Python 3.14 (or a compatible Python 3 version)
- An OpenAI API key with available billing/quota

Open PowerShell and move to the project directory:

```powershell
cd "C:\Users\Ashwini Wandre\OneDrive\Desktop\Agentic_3_0\Smart_agent\Smart_Agent"
```

## 2. Create an isolated environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Configure OpenAI

Copy the safe template, then edit `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

The resulting file should contain:

```env
OPENAI_API_KEY=your_openai_api_key
PLANNER_MODEL=gpt-4o-mini
AGENT_MODEL=gpt-4o-mini
```

The model settings are optional. Never place a real key in source code or commit `.env`.

Confirm that the key is visible without printing it:

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('OPENAI_API_KEY')))"
```

The output should be `True`.

## 4. Verify the installation

Run the complete deterministic suite:

```powershell
python -m pytest
```

Expected result: all tests pass. These tests use scripted model decisions and do not consume OpenAI quota.

Run an API-free graph check:

```powershell
python scripts/run_agent.py --offline --customer-id CUST0003 --conversation-id offline-check --message "Why is my bill higher this month?"
```

Offline mode verifies graph wiring but deliberately executes no business action.

## 5. Run the live agent

Billing:

```powershell
python scripts/run_agent.py --customer-id CUST0003 --conversation-id billing-demo --message "What is included in my plan and why is my monthly bill 599?"
```

Technical support:

```powershell
python scripts/run_agent.py --customer-id CUST0116 --conversation-id technical-demo --message "My router has a red internet light and restarting did not help."
```

Retention:

```powershell
python scripts/run_agent.py --customer-id CUST0058 --conversation-id retention-demo --message "I want to cancel because the service is too expensive."
```

Large refund requiring human approval:

```powershell
python scripts/run_agent.py --interactive-approval --customer-id CUST0094 --conversation-id refund-demo --message "Refund 10000 because I was charged incorrectly."
```

When the business-rule engine confirms that approval is required, the CLI
shows the customer, amount, and reason and asks:

```text
Approve this refund? [y/N]:
```

`yes` executes the refund with a trusted approval flag and returns the outcome
to the specialist loop. `no` leaves the refund unexecuted and records human
escalation. The LLM cannot supply or bypass this decision.

The JSON output shows:

- planner route and routing reason;
- model-selected tool calls and structured outcomes;
- final action;
- KB citations;
- customer-facing response.

The first KB search builds the in-memory FAISS index and calls `text-embedding-3-small`.

## 6. Run the dashboard

Open a second activated PowerShell terminal:

```powershell
cd "C:\Users\Ashwini Wandre\OneDrive\Desktop\Agentic_3_0\Smart_agent\Smart_Agent"
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## 7. Runtime files

- `artifacts/logs/tool_calls.csv` — required tool-call telemetry columns.
- `artifacts/logs/conversation_logs.json` — every graph request.
- `artifacts/logs/mock_tools.jsonl` — detailed tool input/output audit.
- `artifacts/memory/conversation_memory.json` — last 10 interactions per conversation.

Do not add runtime memory or `.env` to the submission ZIP.

## 8. Troubleshooting

### `429 insufficient_quota`

The API key is recognized, but its OpenAI account has no available quota. Restore quota for that account or use another API key with available quota. The graph records the failure and returns a safe human-support response.

### `OPENAI_API_KEY` is missing

Confirm that `.env` is in the repository root and that the virtual environment is active.

### No KB citations

Check the tool trace for `search_kb`. Policy and troubleshooting answers without citations are rejected by the output validator.

### Dashboard has no rows

Run at least one agent or tool request first, then click **Refresh data** in the dashboard.

### Start with clean demo state

Back up logs if needed, then remove generated files under `artifacts/memory/` and the generated rows in `artifacts/logs/`. Keep the CSV header and valid empty JSON structure.
