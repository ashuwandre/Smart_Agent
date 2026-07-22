# Understand the Customer Operations Agent

This document is an interview-preparation guide for the complete repository.
It explains what the project does, how every major layer works, why each
technology was selected, important tradeoffs, alternatives, limitations, and
probable interview questions.

## 1. Thirty-second project explanation

This project is an autonomous Customer Operations Agent for a subscription
business. It accepts billing, refund, technical, and cancellation requests.
An OpenAI planner chooses the correct specialist route. A specialist model then
selects tools such as customer lookup, knowledge-base search, refund, retention
offer, ticket creation, or human escalation.

The LLM does not have final authority over sensitive actions. Pydantic schemas,
guardrails, and a deterministic BusinessRuleEngine validate every tool call.
Refunds above ₹1,000 require trusted human approval. Retention offers require an
eligible customer, cannot exceed 20%, and share a ₹10,000 run budget. Policy and
troubleshooting responses are grounded with FAISS retrieval and KB citations.
JSON memory supports multi-turn conversations, and CSV/JSON logs feed a
Streamlit operations dashboard.

## 2. Main business problem

The business wants to reduce support cost while protecting customers and
revenue. The agent must:

1. understand the customer intent;
2. retrieve trusted customer and policy context;
3. select an appropriate action;
4. execute only actions allowed by business rules;
5. provide a grounded response or escalate safely;
6. preserve multi-turn context;
7. record enough telemetry for operations and audits.

The design optimizes for controlled automation, not maximum automation.

## 3. Action space

The agent has four routes:

- `billing`: invoices, charges, plans, entitlements, and account questions;
- `refund`: requests to return money;
- `technical`: router, Wi-Fi, speed, password, network, and playback issues;
- `retention`: cancellation requests and churn threats.

The agent can select six tools:

- `get_customer(customer_id)`;
- `search_kb(query)`;
- `grant_retention_offer(customer_id, pct)`;
- `issue_refund(customer_id, amount, reason)`;
- `create_ticket(customer_id, summary)`;
- `escalate_to_human(reason, customer_id)`.

Possible final actions are:

- `answer`;
- `clarify`;
- `refuse`;
- `escalate`.

## 4. High-level architecture

```text
Customer message
      |
      v
Load JSON memory
      |
      v
Input guardrails ---- blocked input ----> Safe refusal
      |
      v
LLM planner
      |
      +---- provider failure -----------> Safe escalation
      |
      v
Billing / Refund / Technical / Retention specialist
      |
      v
Bounded LLM tool loop
      |
      +--> strict tool validation
      +--> customer isolation
      +--> call limit and dead-loop detection
      +--> BusinessRuleEngine
      +--> mock business tool or FAISS RAG
      +--> trusted CLI human approval when required
      |
      v
Response generator
      |
      v
Output guardrails
      |
      v
Save JSON memory
      |
      v
Customer response + observability logs
```

The top-level flow is implemented in
`src/customer_ops_agent/agent/graph.py`.

## 5. End-to-end request flow

1. `ObservableGraph.invoke()` receives `conversation_id`, `customer_id`, and
   `message`.
2. `load_memory` loads previous safe context.
3. A conversation cannot switch to a different customer ID. This prevents
   cross-customer memory leakage.
4. `input_guardrail` checks prompt injection, abuse, and PII.
5. PII is masked before any model receives the text.
6. Injection or violent threats take the blocked route.
7. Allowed or reviewable input reaches `LLMPlanner`.
8. The planner returns a structured `RouteDecision`.
9. Planner/provider failure takes a safe human-support path.
10. The selected specialist receives only route-appropriate tools.
11. The specialist may select one or more tools.
12. Every tool request is parsed and strictly validated.
13. Cross-customer access, excessive calls, and repeated loops are rejected.
14. `BusinessRuleEngine.execute_action()` validates financial and eligibility
    rules before dispatching to a tool.
15. Tool results are masked for email and phone before being returned to the
    model.
16. `search_kb` results become citations.
17. The specialist returns strict `SpecialistDraft` JSON.
18. `response_generator` creates an `AgentOutput`.
19. `output_guardrail` checks confidence, citations, secrets, PII, and schema.
20. Both customer and assistant messages are saved atomically.
21. `ObservableGraph` records latency, status, route, confidence, error, and
    circuit-breaker state.

## 6. Why this is one agent rather than five independent agents

The project has one Customer Operations Agent with:

- one planner;
- one shared specialist implementation;
- four route-specific graph nodes;
- route-specific prompts and tool sets.

This avoids unnecessary agent-to-agent communication. Independent agents would
need handoff protocols, conflict resolution, duplicated memory, and more model
calls. The selected architecture preserves specialization while keeping one
authoritative state and one terminal response path.

## 7. LangGraph orchestration

File: `src/customer_ops_agent/agent/graph.py`

Important nodes:

- `_load_memory`: loads prior context and validates customer identity.
- `_input_guardrail`: evaluates and sanitizes inbound text.
- `_input_decision`: selects blocked or continue.
- `_planner_node`: adds safe conversation history and invokes the planner.
- `_planner_decision`: routes to a specialist or planner-failure node.
- `_specialist_node`: adapts the shared specialist to a route-specific node.
- `_blocked_response`: produces a safe refusal without exposing detector rules.
- `_planner_failure_response`: produces safe human escalation.
- `_response_generator`: adds citations and builds structured output.
- `_output_guardrail`: validates the final candidate and fails closed.
- `_save_memory`: stores the turn, offer/refund state, and technical steps.
- `build_graph`: wires and compiles all nodes and wraps them with observability.

Why LangGraph:

- explicit, inspectable state transitions;
- conditional safety routes;
- easier unit testing of each node;
- dependency injection for planners, handlers, memory, and logs;
- clearer than one very large prompt.

Tradeoff:

- more orchestration code;
- planner plus specialist can require multiple model calls.

Alternatives:

- one ReAct agent with all tools: simpler, but less explicit routing;
- OpenAI Agents SDK: useful hosted abstractions, but less aligned with the
  assignment's explicit LangGraph requirement;
- custom Python state machine: fewer dependencies, but more orchestration code.

## 8. Planner logic

File: `src/customer_ops_agent/agent/planner.py`

`RouteDecision` is a Pydantic model containing:

- `route`;
- a short `reason`.

`LLMPlanner`:

- uses `PLANNER_MODEL`, defaulting to `gpt-4o-mini`;
- uses OpenAI structured parsing;
- can return only the four valid routes;
- does not execute tools.

`MockPlanner`:

- is used only for offline checks and deterministic tests;
- routes using keywords from the current message;
- is not the production agency path.

Why structured planner output:

- no fragile string parsing;
- invalid routes are rejected by Pydantic;
- the routing reason is directly observable.

## 9. Specialist and real tool-calling agency

File: `src/customer_ops_agent/agent/handlers.py`

`LLMToolCallingHandler` is the core real-agency component.

Important limits:

- maximum model rounds: 6;
- maximum tool calls: 8;
- route-specific tool lists;
- temperature: 0 for more repeatable decisions.

`_messages()`:

- constructs the specialist system instructions;
- adds safe prior messages;
- adds an abuse-boundary instruction for reviewable abusive language;
- tells the model not to invent facts, policy, eligibility, or action outcomes;
- requires KB search for policy and troubleshooting;
- tells the model to clarify ambiguous requests.

`_completion()`:

- calls the specialist model;
- supplies route-specific tool schemas;
- requires strict `SpecialistDraft` JSON;
- retries transient provider errors up to three times;
- uses a separate circuit breaker for each route.

`_execute_tool()`:

1. parses model-provided JSON arguments;
2. rejects non-object arguments;
3. normalizes a denied-refund escalation reason;
4. rejects access to a customer other than the active conversation customer;
5. reserves one tool-call slot;
6. checks repeated tool/argument combinations;
7. invokes the BusinessRuleEngine;
8. invokes the trusted human callback when required;
9. produces a structured `ToolCallTrace`.

`__call__()`:

- runs the bounded model/tool loop;
- feeds tool results back to the model;
- gathers KB citations;
- safely handles model or parsing failures;
- guarantees escalation when approval is required or denied;
- returns `HandlerResult`.

Why this is genuine agency:

- the live model chooses which allowed tools to call;
- the model may call multiple tools and react to their results;
- the output is not selected from hardcoded customer-message responses;
- deterministic code constrains unsafe actions without replacing model choice.

## 10. Human-in-the-loop refund flow

Relevant files:

- `scripts/run_agent.py`;
- `src/customer_ops_agent/agent/handlers.py`;
- `src/customer_ops_agent/business_rules/engine.py`;
- `src/customer_ops_agent/guardrails/tools.py`;
- `src/customer_ops_agent/tools/mock_tools.py`.

Flow for refund above ₹1,000:

1. The LLM selects `issue_refund`.
2. The BusinessRuleEngine returns `requires_human_approval`.
3. The refund remains unexecuted.
4. With `--interactive-approval`, the handler calls the trusted CLI callback.
5. The CLI shows customer, amount, and reason.

If the human enters `yes`:

1. trusted code calls `execute_action(..., human_approved=True)`;
2. the action is revalidated;
3. the refund tool executes;
4. a `RFND######` transaction ID is generated;
5. the tool result returns to the model;
6. the specialist completes the response.

If the human enters `no`:

1. the original refund remains unexecuted;
2. trace status becomes `human_approval_denied`;
3. no refund transaction ID is created;
4. `escalate_to_human` records an `ESC######` ID;
5. the final response clearly says approval was denied.

Security property:

- `human_approved` is intentionally absent from the LLM-facing refund schema;
- unknown fields are forbidden;
- only trusted application code can supply approval.

The CLI is only an input/output interface. It does not hardcode the agent's
route, tools, or outcome.

## 11. BusinessRuleEngine

File: `src/customer_ops_agent/business_rules/engine.py`

Authoritative constants:

- refund approval threshold: ₹1,000;
- maximum offer: 20%;
- total offer budget: ₹10,000 per engine run;
- minimum tenure: 12 months;
- minimum churn risk: 0.70;
- offer duration: 3 months.

Offer cost:

```text
monthly fee × offer percentage / 100 × 3 months
```

`validate_action()`:

- validates the tool name and arguments first;
- returns `requires_human_approval` for a large unapproved refund;
- delegates retention requests to `_validate_retention_offer`;
- returns structured pass/fail reasons.

`_validate_retention_offer()` checks:

- customer exists;
- percentage does not exceed 20%;
- tenure is at least 12 months;
- churn risk is at least 0.70;
- account is active;
- no outstanding invoice;
- customer has not received another offer in the run;
- offer cost fits the remaining budget.

Budget reservation uses a lock. This prevents concurrent actions from both
seeing the same available budget and overspending.

`execute_action()`:

- validates before dispatch;
- logs rejected actions;
- executes only approved tools;
- passes human approval to the refund tool only from trusted code;
- returns `ActionExecutionResult`.

Why rules are code rather than prompt instructions:

- prompts are probabilistic;
- financial controls must be deterministic;
- rule outcomes need structured reasons and tests;
- concurrency-safe budget reservation cannot be guaranteed by an LLM.

## 12. Tool validation and tool implementations

Tool validation file: `src/customer_ops_agent/guardrails/tools.py`

Validation uses strict Pydantic models:

- unknown tools are rejected;
- unknown fields are rejected;
- customer IDs match `CUST####`;
- offer percentage must be greater than 0 and at most 20;
- refund amount must be positive;
- strings have length limits.

Mock tool file: `src/customer_ops_agent/tools/mock_tools.py`

`get_customer()`:

- loads and validates synthetic customer records;
- normalizes customer ID;
- returns a typed customer record or error.

`search_kb()`:

- builds the RAG index on the first request;
- reuses it for later requests in the process;
- returns the top three chunks.

`grant_retention_offer()`:

- repeats critical eligibility and budget checks as defense in depth;
- calculates offer cost using `Decimal`;
- prevents duplicate offers;
- returns remaining budget.

`issue_refund()`:

- validates customer, amount, and reason;
- returns pending approval when required;
- issues only with valid trusted approval;
- returns a deterministic transaction ID.

`create_ticket()`:

- validates customer and summary;
- returns `TKT######`.

`escalate_to_human()`:

- validates reason and optional customer;
- returns `ESC######`.

All tools use `@audit_tool`.

## 13. Tool audit logging

File: `src/customer_ops_agent/tools/audit.py`

`tool_telemetry_context()` stores orchestration metadata such as:

- confidence;
- retry count;
- loop count.

`audit_tool`:

- records start time;
- safely captures input;
- executes the tool;
- captures output or exception;
- calculates latency;
- writes JSONL audit data;
- writes the dashboard CSV event.

Why a decorator:

- logging behavior is consistent;
- individual tool implementations remain focused;
- new tools can gain telemetry with one annotation.

## 14. RAG and knowledge-base grounding

File: `src/customer_ops_agent/rag/store.py`

Source:

- ten Markdown documents in `data/knowledge_base`.

Chunking:

- paragraph-aware;
- target chunk size: 1,200 characters;
- overlap: 200 characters;
- document title repeated on later chunks.

Metadata:

- filename;
- chunk ID;
- title.

Embeddings:

- model: `text-embedding-3-small`;
- vectors are converted to `float32`;
- vectors are L2 normalized.

FAISS:

- uses `IndexFlatIP`;
- inner product on normalized vectors equals cosine similarity;
- search returns the top three chunks.

`build_index()`:

- reads sorted Markdown files;
- chunks and embeds text;
- builds a new index locally;
- swaps index and records together under a lock;
- does not expose a partially built index.

`search()`:

- rejects an empty query;
- embeds the query;
- verifies vector dimensions;
- returns content, metadata, and score.

Why FAISS:

- fast local similarity search;
- no external vector database;
- appropriate for ten assignment documents;
- simple and inexpensive.

Alternatives:

- pgvector for durable shared production retrieval;
- Pinecone or another managed vector database for scaling;
- BM25 for lexical retrieval;
- hybrid BM25 plus vectors for better exact-term recall.

Limitation:

- the index is in memory and rebuilt after process restart.

## 15. Input guardrails

File: `src/customer_ops_agent/guardrails/input.py`

`detect_prompt_injection()` checks:

- instruction override;
- system-prompt exfiltration;
- role impersonation;
- guardrail/control bypass.

`detect_abuse()` distinguishes:

- violent threats;
- ordinary abusive language.

`mask_pii()` masks:

- email;
- payment card;
- Aadhaar number;
- Indian phone number.

`evaluate_input()` returns:

- `BLOCK` for strong injection or violent threat;
- `REVIEW` for ordinary abuse;
- `ALLOW` otherwise.

Reviewable abuse is not ignored. The specialist receives an instruction to set
a respectful boundary and still address a legitimate request.

Tradeoff:

- regex is deterministic, inexpensive, and testable;
- it may have false positives or miss novel attacks.

Production alternative:

- layered regex, classifier model, policy service, and adversarial monitoring.

## 16. Execution guardrails

File: `src/customer_ops_agent/guardrails/execution.py`

`ToolCallLimiter`:

- maximum eight calls per request;
- reserves a call before execution;
- rejects later requests after the limit.

`DeadLoopDetector`:

- serializes node, tool, and arguments;
- hashes the value so sensitive arguments are not retained;
- examines a rolling window of ten;
- flags the third identical occurrence.

`execute_with_retry()`:

- retries only declared transient errors;
- uses exponential backoff;
- does not retry business or validation failures.

`CircuitBreaker`:

- begins `CLOSED`;
- changes to `OPEN` after three failures;
- rejects calls during the 30-second cooldown;
- changes to `HALF_OPEN` for one recovery probe;
- returns to `CLOSED` after success.

Why both retry and circuit breaker:

- retry handles isolated transient errors;
- circuit breaker prevents repeated calls to an unhealthy dependency.

## 17. Output guardrails and confidence

File: `src/customer_ops_agent/guardrails/output.py`

`check_confidence()`:

- default threshold is 0.70;
- accepted output continues;
- low confidence requires escalation.

`validate_output()` checks:

- valid `AgentOutput` schema;
- confidence threshold;
- citations when `policy_based=True`;
- empty citation values;
- possible secret/token patterns;
- PII in customer-facing response.

If validation fails, the graph returns a generic safe fallback.

What confidence means:

- it is the specialist model's self-reported certainty from 0 to 1;
- it is not a statistically calibrated probability;
- production thresholds require validation against human-reviewed outcomes.

Why dashboard confidence can be null:

- planner errors occur before a specialist output exists;
- no specialist output means no confidence value exists;
- tool events may also lack confidence if it was not available in telemetry
  context at the moment of execution.

Null therefore means "not available," not zero confidence.

## 18. Conversation memory

Files:

- `src/customer_ops_agent/memory/models.py`;
- `src/customer_ops_agent/memory/store.py`.

Storage:

- `artifacts/memory/conversation_memory.json`;
- one local JSON document;
- atomic `.tmp` write followed by replace;
- protected by `RLock`.

Remembered data:

- last ten customer/assistant interactions;
- customer ID;
- most recent offer;
- most recent refund request/status;
- last ten technical steps.

Safety:

- one conversation cannot change customer ID;
- blocked hostile inputs are stored for continuity but excluded from future
  model context;
- invalid/corrupted JSON fails closed.

Why JSON:

- assignment explicitly requires no database;
- easy to inspect and demonstrate;
- sufficient for one local process.

Production alternatives:

- PostgreSQL for durable transactions;
- Redis for low-latency session context;
- encrypted event store for audit history.

## 19. Observability

Files:

- `src/customer_ops_agent/observability/models.py`;
- `src/customer_ops_agent/observability/store.py`;
- `src/customer_ops_agent/observability/graph.py`.

Artifacts:

- `artifacts/logs/tool_calls.csv`;
- `artifacts/logs/conversation_logs.json`;
- `artifacts/logs/mock_tools.jsonl`.

`ObservabilityStore`:

- creates stable files and schemas;
- appends tool CSV events;
- atomically updates conversation JSON;
- retains up to 5,000 conversation events.

`ObservableGraph`:

- delegates to the compiled graph;
- logs successful, blocked, provider-failed, and raised requests;
- sanitizes logged request text;
- supports synchronous and asynchronous invocation.

Why a graph wrapper:

- request telemetry is guaranteed independently of graph node behavior;
- exceptions are logged before they are re-raised;
- the compiled graph interface remains accessible through `__getattr__`.

## 20. Streamlit dashboard

File: `dashboard/app.py`

The dashboard displays:

- remaining retention budget;
- circuit-breaker status;
- human approval queue;
- average tool latency;
- errors and denials;
- tool timeline;
- latency by tool.

The dashboard is read-only. It consumes local telemetry and does not modify
agent decisions.

Why Streamlit:

- minimal UI code;
- good for local data demonstrations;
- automatic table and chart rendering.

Production alternatives:

- Grafana with Prometheus/OpenTelemetry;
- Datadog;
- custom operations console with RBAC.

## 21. Data

`data/synthetic/customers.json`:

- exactly 200 customers;
- plans, monthly fees, tenure, region, contact details, churn score, invoices,
  entitlements, services, and status.

`data/synthetic/inbox.json`:

- 25 inbound messages;
- billing, cancellation, refund, technical, abuse, injection, ambiguity,
  upgrade, password, and network examples.

`data/knowledge_base`:

- ten Markdown policy and troubleshooting documents.

`data/synthetic/demo_scenarios.json`:

- structured scenarios used by integration tests.

## 22. Testing strategy

Run:

```powershell
python -m pytest
```

Current result:

- 45 tests pass;
- 12 integration test cases across 11 scenario categories;
- unit coverage for graph, tools, business rules, memory, guardrails,
  observability, RAG, and HITL.

Why scripted LLM tests are acceptable:

- tests must be deterministic;
- CI should not depend on provider quota;
- scripted responses test the real handler loop and control code;
- live execution still uses the OpenAI planner and tool selection.

The scripted test responses are test doubles, not production hardcoding.

## 23. Red-flag compliance

Hardcoded outputs:

- production does not match exact customer messages to canned responses;
- only offline/test doubles are deterministic.

Single monolithic prompt:

- not used;
- LangGraph separates memory, guardrails, planning, specialists, output, and
  persistence.

No real tool calling:

- false;
- the live specialist receives tool schemas, chooses tools, and reacts to their
  results.

Ignored budgets or thresholds:

- false;
- business rules enforce retention budget and refund HITL in code.

Missing guardrails:

- false;
- controls exist before the model, before tools, during execution, and before
  output.

Not runnable:

- tests, offline CLI, live CLI, PDF generator, and dashboard have documented
  commands;
- live model execution requires valid API access and available quota.

## 24. Important limitations

- CRM, billing, refund, ticket, and escalation systems are mocks.
- Memory and logs are local files, not shared distributed stores.
- Human approval uses CLI input, not a durable approval service or web UI.
- FAISS index is rebuilt in each process.
- Input regex cannot guarantee detection of every novel prompt injection.
- Model confidence is self-reported.
- There is no idempotency key for repeated external financial actions.
- There is no formal Tier 3 statistical scorecard.
- Mock sequence IDs restart with each process.

These limitations should be stated honestly in an interview.

## 25. Production plan

1. Replace mocks with sandbox adapters using the same Pydantic contracts.
2. Add authentication, RBAC, secrets management, and encryption.
3. Add idempotency keys for refunds, offers, and tickets.
4. Store conversations, approvals, budgets, and audits in durable databases.
5. Run in shadow mode with all actions disabled.
6. Compare proposals against human decisions.
7. Canary low-risk billing and troubleshooting requests.
8. Keep refunds and offers approval-only initially.
9. Add OpenTelemetry and centralized metrics/alerts.
10. Calibrate confidence and retrieval thresholds.
11. Expand only when grounding, safety, cost, latency, and business KPIs meet
    agreed thresholds.

## 26. Cost considerations

Variable costs:

- planner tokens;
- specialist tokens across multiple rounds;
- embedding API calls;
- future external CRM/ticketing calls;
- human escalation and approval effort;
- retention discount cost.

Cost controls:

- cheaper planner/specialist default model;
- route-specific tools reduce prompt size;
- maximum rounds and tool calls;
- lazy KB indexing;
- cache embeddings in production;
- fail fast on unsafe input;
- use deterministic code rather than extra model calls for rules.

## 27. Business KPIs

Leadership should monitor:

- resolution rate by intent;
- escalation rate by intent;
- net retained monthly revenue minus discount cost;
- retention acceptance rate;
- refund approval and rejection rate;
- cost per resolved request;
- average and p95 latency;
- grounding/citation failure rate;
- incorrect tool-call rate;
- approval-control violations;
- prompt-injection block rate;
- customer satisfaction after automation.

## 28. Common interview questions and answers

### Q1. What is agentic about this project?

The live specialist model dynamically selects from route-specific tools, may
call several tools, observes their results, and decides whether to answer,
clarify, refuse, or escalate. Deterministic controls constrain the action but do
not replace model tool selection.

### Q2. Why did you use LangGraph?

It provides explicit stateful orchestration, conditional routing, and testable
nodes. That makes blocked inputs, provider failures, specialist routes, output
validation, and memory persistence visible rather than hidden in one prompt.

### Q3. Why not use five agents?

One orchestrated agent avoids duplicated memory, conflicting decisions, and
handoff complexity. Four specialist nodes still provide route-specific prompts
and tool access.

### Q4. Why use an LLM planner instead of keyword routing?

Customer language is varied and contextual. Structured LLM routing generalizes
better. Keyword routing exists only as an offline/test double.

### Q5. Can the LLM bypass the refund approval?

No. `human_approved` is excluded from its tool schema, extra fields are
forbidden, and only the trusted callback can call the engine with approval.

### Q6. What happens when a human approves?

The engine revalidates the same action with trusted approval, executes the
refund tool, returns a transaction ID to the specialist, and the specialist
completes the response.

### Q7. What happens when a human rejects?

The refund remains unexecuted, status becomes `human_approval_denied`, and the
case is escalated with a reason reflecting the denial.

### Q8. How do you stop retention overspending?

The engine calculates three-month offer cost with `Decimal`, validates all
eligibility rules, and atomically reserves budget under a lock. Total reserved
cost cannot exceed ₹10,000.

### Q9. Why are business rules duplicated in the engine and mock tools?

The engine is the authoritative pre-execution gate. Tool checks provide defense
in depth and simulate downstream-system validation. In production, shared
policy configuration should prevent constants from drifting.

### Q10. How do you prevent hallucinated policy?

The specialist is instructed to call `search_kb`; retrieved chunks generate
citations; the output validator rejects policy-based answers without citations.
Action confirmations are grounded by structured tool results.

### Q11. Why use FAISS?

The KB is small and local. FAISS provides fast cosine retrieval without an
external database. A production system could use pgvector or a managed vector
database.

### Q12. Why normalize embeddings?

With unit-length vectors, inner product in `IndexFlatIP` equals cosine
similarity. Larger scores represent closer semantic matches.

### Q13. How is customer data protected?

Inbound PII is masked before model use. Tool results redact email and phone.
Cross-customer tool calls and memory reuse are rejected.

### Q14. How do you handle prompt injection?

High-precision patterns run before the planner. Strong injection is blocked.
Even if text reaches a specialist, strict tools, business rules, and output
validation limit impact.

### Q15. Why not rely only on the system prompt?

Prompts are not enforcement. Tool allowlists, strict schemas, customer
isolation, financial rules, call limits, loops, and approval are enforced in
deterministic code.

### Q16. What is the difference between retry and circuit breaker?

Retry handles short transient failures. Circuit breaker stops repeated calls to
an unhealthy dependency and allows one controlled recovery probe later.

### Q17. Why is confidence sometimes null in the dashboard?

If planning fails before the specialist produces output, no confidence score
exists. Null means unavailable, not zero.

### Q18. Why is model confidence a limitation?

It is self-reported and not calibrated against observed correctness. Production
thresholds need evaluation with labeled human-reviewed examples.

### Q19. How does memory avoid leakage?

A conversation is bound to one customer ID. Customer mismatch raises an error.
Blocked hostile input is excluded from future planner context.

### Q20. Why atomic JSON writes?

Writing a complete temporary file and replacing the original prevents readers
from seeing a partially written document.

### Q21. Why keep only ten interactions?

It satisfies the requirement, bounds token usage and storage growth, and limits
stale context.

### Q22. How does observability work?

The tool decorator records tool-level JSONL and CSV events. `ObservableGraph`
records every conversation outcome, including failures. Streamlit reads those
files.

### Q23. Does CLI-based HITL mean outputs are hardcoded?

No. CLI is a transport and trusted approval surface. The model still selects
the route and tool; the human supplies only a yes/no authorization decision.

### Q24. How do you test agency without spending API quota?

Scripted model clients return controlled tool-call messages. The real handler,
schemas, business engine, tool loop, and escalation logic execute normally.

### Q25. What would you change first for production?

Add durable transactional stores and idempotency keys, connect sandbox business
systems, implement authenticated approval workflows, and run in shadow mode.

### Q26. What is the most important safety principle?

The LLM may propose an action, but deterministic code decides whether that
action executes.

## 29. Commands

Setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Tests:

```powershell
python -m pytest
```

Live request:

```powershell
python scripts/run_agent.py --customer-id CUST0003 --conversation-id demo-billing --message "Why is my bill higher?"
```

HITL request:

```powershell
python scripts/run_agent.py --customer-id CUST0094 --conversation-id demo-refund --message "Refund 10000" --interactive-approval
```

Offline structure check:

```powershell
python scripts/run_agent.py --offline --customer-id CUST0003 --message "Why is my bill higher?"
```

Dashboard:

```powershell
streamlit run dashboard/app.py
```

Regenerate stakeholder PDF:

```powershell
python scripts/generate_stakeholder_pdf.py
```

Detailed recording cases:

- `docs/DEMO_RECORDING_GUIDE.md`.

## 30. Files to remember for an interview

- `src/customer_ops_agent/agent/graph.py`: graph and control flow.
- `src/customer_ops_agent/agent/planner.py`: structured routing.
- `src/customer_ops_agent/agent/handlers.py`: real tool loop and HITL.
- `src/customer_ops_agent/business_rules/engine.py`: financial rules.
- `src/customer_ops_agent/guardrails/input.py`: injection, abuse, PII.
- `src/customer_ops_agent/guardrails/output.py`: confidence and grounding.
- `src/customer_ops_agent/guardrails/tools.py`: strict tool schemas.
- `src/customer_ops_agent/guardrails/execution.py`: resilience controls.
- `src/customer_ops_agent/rag/store.py`: embeddings and FAISS.
- `src/customer_ops_agent/tools/mock_tools.py`: simulated business systems.
- `src/customer_ops_agent/memory/store.py`: bounded JSON memory.
- `src/customer_ops_agent/observability/graph.py`: request logging.
- `dashboard/app.py`: operations view.
- `scripts/run_agent.py`: CLI and trusted approval callback.

## 31. Final interview summary

The strongest design decision is separation of probabilistic reasoning from
deterministic authority:

- the LLM understands intent and selects tools;
- RAG supplies policy context;
- Pydantic validates contracts;
- guardrails constrain unsafe behavior;
- the BusinessRuleEngine controls financial actions;
- the human controls high-impact refunds;
- memory supplies bounded context;
- observability makes every outcome reviewable.

This design provides real agency without giving the model unrestricted control.
