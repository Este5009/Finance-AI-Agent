# AGENTS.md

## Initial Instructions

Before any task, read:

1. `AGENTS.md`
2. `MEMORY.md`

Update `MEMORY.md` whenever a durable architecture decision, coding preference, naming convention, or workflow rule is learned.

Do not commit `AGENTS.md` or `MEMORY.md` unless the user explicitly asks for
repository policy, documentation, or durable workflow-rule changes.

---

## Project Objective

Build an autonomous AI-powered financial analyst.

This is not a chatbot. The system should ingest financial reports, normalize messy formats, calculate financial metrics, detect anomalies, investigate issues, use memory/history, and generate strategic reports.

---

## Target Pipeline

Integrated Excel workbook (`.xlsx` / `.xls`) containing financial actuals, budgets, goals/targets, variances, departments, payroll, collections, and cash-flow data

↓

Document ingestion

↓

Flexible schema normalization

↓

Deterministic finance calculations + anomaly detection

↓

Agent planner/orchestrator

↓

Historical memory + retrieval tools

↓

LLM analysis with Ollama

↓

PDF / HTML / Excel / JSON report outputs

↓

Simple Streamlit interface

---

## Core Principle

Python is the source of truth.

Python handles:

- File extraction
- Data cleaning
- Schema normalization when possible
- Financial calculations
- KPI computation
- Validation
- Rule-based anomaly detection
- Statistical anomaly detection
- Historical storage
- Report generation

The LLM handles:

- Structure interpretation only when deterministic mapping fails
- Investigation planning
- Financial explanations
- Strategic recommendations
- Risk interpretation
- Executive report wording

Never use the LLM as the source of truth for math.

---

## Ollama Usage

Default local LLM:

- Ollama
- `qwen3:30b-a3b`

Use Ollama for two main roles:   

1. Structure Interpreter  
   Maps messy or Spanish report formats into the standard schema when deterministic mapping is insufficient.

2. Financial Analyst  
   Explains calculated results, anomalies, risks, trends, and recommendations.

Ollama should receive structured summaries, not entire files by default.

---

## Historical Access Strategy

Store full historical reports and processed results.

Do not send all historical data to the LLM by default.

Use retrieval tools:

- `get_previous_cycle_memory()`
- `get_department_history()`
- `get_vendor_history()`
- `get_payroll_history()`
- `get_transactions()`
- `get_full_report()`

The agent should start with summaries and request deeper data only when needed.

---

## Agentic Behavior

The agent should not only run a fixed pipeline.

It should investigate.

Example:

1. Detect Engineering overspending.
2. Check department history.
3. Check vendor payments.
4. Check payroll trends.
5. Check prior recommendations.
6. Escalate risk if repeated.
7. Generate strategy and actions.

The LLM orchestrates investigation. Python tools execute the work.

---

## Development Order

Build in phases:

1. Processing engine
2. Schema normalization
3. Finance calculations
4. Anomaly detection
5. Historical storage and memory tools
6. Ollama integration
7. Agent orchestrator
8. PDF / Excel outputs
9. Streamlit interface

Do not jump ahead unless instructed.

---

## Code Quality

Write readable, modular Python.

Every function must have a docstring explaining:

- Purpose
- Inputs
- Outputs
- Assumptions

Use frequent inline comments explaining both:

- Why code exists
- How technical logic works

Prefer clarity over cleverness.

---

## Development Workflow

Before coding:

- Understand existing architecture.
- Reuse existing modules.
- Avoid duplicate logic.
- Keep deterministic tools separate from LLM logic.

After coding, summarize:

- Files modified
- What changed
- Why it changed
- Assumptions
- Risks or next steps

---

## Persistent Process and Timeout Policy

Codex must never directly launch or wait on persistent local services as part
of normal task completion.

Mandatory rules:

- Codex must never launch Streamlit, Ollama, dev servers, watchers, browser
  sessions, or background workers as part of normal task completion.
- Codex must never call `Start-Process` for a persistent service.
- Codex must never wait for a server process to exit.
- Server startup and interactive browser testing are user-owned actions unless
  explicitly requested.
- Validate Streamlit changes using:
  - imports
  - Streamlit AppTest when available
  - helper/unit tests
  - report-model/artifact rendering tests
- Validate server availability only when it is already running, using a bounded
  HTTP check of at most 10 seconds.
- If no server is running, report the manual launch command instead of starting
  it.
- Every non-model shell command must have a bounded timeout.
- Focused tests: maximum 5 minutes.
- Full test suite: maximum 10 minutes.
- Ordinary inspection commands: maximum 30 seconds.
- Never rerun an expensive Ollama pipeline unless explicitly required.
- Reuse existing artifacts for presentation/report tests.
- A failed optional health check must not prevent committing and finishing the
  task.
- Complete code, tests, commit, and push before any optional environment check.
- Never stage `outputs/`, databases, uploaded files, caches, or generated
  financial artifacts.

Manual launch helpers are provided for users:

- Windows: `.\scripts\start_streamlit_windows.ps1`
- macOS/Linux: `bash scripts/start_streamlit_macos.sh`

Codex may run `scripts/check_local_services.py` for bounded availability
checks, but only as a non-blocking diagnostic.

---

## Task Completion Checklist

For every future Codex task:

1. Inspect current diff.
2. Implement the requested change.
3. Run focused bounded tests.
4. Run full bounded tests.
5. Validate with existing artifacts where possible.
6. Commit only source/tests/configuration.
7. Push.
8. Report the manual Streamlit command.
9. Stop.

Starting Streamlit is not part of Codex task completion.

---

## Git Rules

Never commit:

- API keys
- Secrets
- `.env`
- Personal AI tool config files

Commit `AGENTS.md` or `MEMORY.md` only when the user explicitly asks for
repository policy, documentation, or durable workflow-rule changes.

Check `git status` before and after major changes.
