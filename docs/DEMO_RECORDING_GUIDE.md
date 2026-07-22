# Demo Recording Guide

Use these five live cases to demonstrate routing, genuine tool selection,
knowledge-base grounding, business rules, and both human-approval outcomes.

## Before recording

1. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Confirm `.env` contains an OpenAI API key with available API quota.
3. Start the dashboard in a second terminal:

   ```powershell
   streamlit run dashboard/app.py
   ```

4. Keep the CLI and dashboard visible. After each case, refresh the dashboard
   and briefly show the Tool Timeline, Human Approval Queue, latency, and errors.
5. Use the unique conversation IDs below. This prevents earlier memory from
   changing the planner's routing reason during the recording.

## Case 1: Billing answer grounded in the KB

Run:

```powershell
python scripts/run_agent.py --customer-id CUST0003 --conversation-id demo-billing-01 --message "Why is my monthly bill 599, and what does my current plan include?"
```

Show:

- route is `billing`;
- `get_customer` retrieves trusted account and plan facts;
- `search_kb` retrieves billing or plan policy;
- citations contain KB filename and chunk ID;
- the response does not invent account facts.

Typical tools: `get_customer`, `search_kb`.

## Case 2: Technical problem and support ticket

Run:

```powershell
python scripts/run_agent.py --customer-id CUST0116 --conversation-id demo-technical-01 --message "My router has a red light. I restarted it and checked the cables, but it still does not work. Follow the documented troubleshooting policy and create a support ticket if those completed steps are not enough."
```

Show:

- route is `technical`;
- `search_kb` supplies documented router steps;
- the agent recognizes that the stated steps were already attempted;
- `create_ticket` returns a structured ticket ID when the model determines that
  escalation through a ticket is appropriate;
- the final response includes KB citations for troubleshooting claims.

Typical tools: `get_customer`, `search_kb`, `create_ticket`.

## Case 3: Eligible retention offer with budget control

`CUST0058` is suitable for this demonstration: active account, 56 months of
tenure, churn risk `0.72`, and no outstanding invoice.

Run:

```powershell
python scripts/run_agent.py --customer-id CUST0058 --conversation-id demo-retention-01 --message "The service is too expensive and I want to cancel. Check whether I qualify for a retention offer and, if eligible, grant a 20 percent discount."
```

Show:

- route is `retention`;
- `get_customer` verifies tenure, churn risk, account status, and invoice state;
- `search_kb` grounds the retention policy;
- `grant_retention_offer` is selected by the model;
- the BusinessRuleEngine enforces the 20% cap and ₹10,000 run budget;
- the tool output shows offer cost and remaining budget.

Typical tools: `get_customer`, `search_kb`, `grant_retention_offer`.

## Case 4: Large refund approved by a human

Run:

```powershell
python scripts/run_agent.py --customer-id CUST0094 --conversation-id demo-refund-approved-01 --message "Please refund 10000 for a verified duplicate charge." --interactive-approval
```

When prompted, enter:

```text
y
```

Show:

- route is `refund`;
- the initial action is blocked because ₹10,000 exceeds ₹1,000;
- the CLI displays the customer, amount, and reason outside model context;
- the trusted `yes` causes the BusinessRuleEngine to revalidate the action;
- `issue_refund` executes and returns `status: issued` plus a transaction ID;
- the agent continues and returns a successful customer-facing response.

Important explanation: the LLM cannot send `human_approved`. That field is
excluded from the model-facing tool schema. Only the trusted CLI callback can
provide the approval decision.

## Case 5: Large refund rejected by a human

Run:

```powershell
python scripts/run_agent.py --customer-id CUST0094 --conversation-id demo-refund-denied-01 --message "Please refund 10000 for a verified duplicate charge." --interactive-approval
```

When prompted, enter:

```text
n
```

Show:

- route is `refund`;
- `issue_refund` has `executed: false`;
- its status is `human_approval_denied`;
- no refund transaction ID is generated;
- `escalate_to_human` executes and returns an escalation ID;
- the reason states that approval was denied, not that approval is still pending;
- the final action is `escalate`.

Typical tools: blocked `issue_refund`, then `escalate_to_human`.

## What each JSON field proves

- `route`: planner selected a specialist.
- `routing_reason`: short explanation of why that specialist was chosen.
- `planner_error`: `null` means planning succeeded.
- `tool_calls`: model-selected tools plus validated arguments and outcomes.
- `executed`: whether deterministic controls permitted the action.
- `status`: issued, granted, created, denied, rejected, or escalated outcome.
- `citations`: KB sources used for policy or troubleshooting claims.
- `action`: final answer, clarification, refusal, or escalation decision.
- `response`: customer-facing output after output guardrails.

## Suggested recording order

1. Spend 20–30 seconds showing the architecture in
   `docs/stakeholder_summary.pdf`.
2. Run Case 1 to prove grounded retrieval.
3. Run Case 3 to prove financial eligibility and budget controls.
4. Run Case 4 and enter `y`.
5. Run Case 5 and enter `n`.
6. If time permits, run Case 2 to show troubleshooting and ticket creation.
7. Finish on the dashboard and show that the same tool actions were logged.

Do not show `.env`, the API key, hidden prompts, payment-card data, or private
customer contact details in the recording.
