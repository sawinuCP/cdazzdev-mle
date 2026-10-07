# Task 3 — Agentic Workflows: Multi-Agent Financial Research System (Revised Plan v2)

> Repo folder: `task3_agentic/` · Max score 100 (+5 dashboard bonus, +5 video) · Est. effort 5–6 h
> Prepared: 2026-10-06 · Supersedes `plan_agentic_research.md`
> Repo-level rules (folder names, CITATIONS.md, REFLECTION.md, no keys, visible outputs) are in `plan_task1_financial.md` §0.

---

## 1. Objective

Given a ticker, autonomous agents produce: a **Financial Health Summary**, **Top-3 90-day risks** with evidence, and **one data-driven hedge**. Delivered in three stages that mirror the rubric:

- **3A:** one agent with all five tools, deciding its own call order.
- **3B:** the same machinery split into Agent A (quant) and Agent B (writer) with enforced tool whitelists, a Pydantic handoff and a critique loop.
- **3C:** short-term memory, persistent ticker+date cache, `agent_trace.jsonl`.

**Success criteria:** zero manual intervention; ≥ 1 visible observe→replan; ≥ 1 visible critique→respond→incorporate; every tool call traced; no unhandled exceptions.

---

## 2. Rubric traceability

| Criterion (pts) | Implemented by | Evidence |
|---|---|---|
| Five tools correct (15) | `tools.py` | unit tests + one demo call each |
| Autonomous tool selection (10) | LLM-chosen JSON actions in a ReAct loop | trace with varying order |
| Observe→replan visible (8) | `replan_reason` field + event | notebook trace |
| Final report, 3 sections, evidence (10) | `ResearchReport` + validators | report JSON/markdown |
| Graceful errors (7) | `ToolResult` contract + fault injection | chaos cell |
| Roles + enforced restriction (8) | whitelist guard | blocked-call demo |
| Structured handoff (8) | `AgentBrief`, `CritiqueRequest`, `ClarificationResponse` | printed JSON |
| Message trace visible (6) | trace printer | notebook output |
| Critique loop (8) | conditional edge, counter = 1 | visible request/response |
| End-to-end automation (5) | `run_research(ticker)` | one-cell run |
| Short-term memory (5) | `MemoryStore` + follow-up | no new tool calls |
| Persistent cache (5) | `cache/{TICKER}_{YYYY-MM-DD}.json` | second run prints cache HIT |
| `agent_trace.jsonl` (5) | `TraceLogger` | committed file |
| Bonus dashboard (+5) | Streamlit | README screenshot |

---

## 3. Stack and layout

**LangGraph for orchestration, plain Python for the ReAct loop.** LangGraph gives a typed state, conditional edges for the critique routing and a visible graph. The inner loop uses a **JSON-action protocol** (the LLM returns `{thought, action, args, replan_reason?}` validated by Pydantic) rather than provider-specific native function calling, so it works on any OpenAI-compatible gateway. Simpler than CrewAI or a full LangChain agent executor, and every step stays visible.

```
task3_agentic/
├── README.md                # Colab badge, architecture picture, dashboard screenshot
├── notebook/task3_agentic.ipynb
├── src/
│   ├── config.py            # constants: loop caps, thresholds, paths
│   ├── tools.py             # five tools + ToolResult
│   ├── indicators.py        # small copy from Task 1 (cited SOURCE: own Task 1 file)
│   ├── schemas.py           # AgentAction, AgentBrief, CritiqueRequest, ClarificationResponse, ResearchReport
│   ├── llm_client.py        # same design as Task 1 (retry, JSON parse, cache)
│   ├── prompts.py           # all prompts
│   ├── tracing.py           # TraceLogger -> logs/agent_trace.jsonl
│   ├── agents.py            # ReAct loop, whitelist guard, Agent A / Agent B runners
│   ├── graph.py             # LangGraph state machine
│   ├── memory.py            # MemoryStore + persistent cache
│   └── main.py              # run_research(ticker)
├── dashboards/trace_dashboard.py
├── logs/agent_trace.jsonl   # committed
├── cache/                   # ticker+date JSON (commit one sample)
├── tests/
└── outputs/                 # report.json, report.md
```
Self-contained: indicator code is copied from Task 1 with a `# SOURCE:` comment, not imported across folders.

---

## 4. The five tools

Contract: every tool returns `ToolResult{ok, data, error, hint, source, fetched_at}` and **never raises**. The agent reads `ok=false` + `hint` and adapts.

1. `get_price_data(ticker, period)` — yfinance OHLCV tail + indicator snapshot (SMA50/200, RSI, MACD, %B, 52w range, YTD).
2. `get_news(ticker, n)` — headline ladder (yfinance → Yahoo RSS → Google RSS), returns `[{title, source, published, url}]`; handle both yfinance news shapes (flat and nested under `content`).
3. `calculate_volatility(ticker, window)` — `std(log_returns, ddof=0) × √252`; window validated 5–252; returns annualized, daily, window.
4. `llm_sentiment(headlines)` — per-headline sentiment + aggregate score via the LLM client, Pydantic-gated; on failure a neutral sentinel with `estimate=true`.
5. `web_search(query)` — `duckduckgo-search` `DDGS().text(query, max_results=6)`; polite sleep + one backoff retry. Note: the package has been renamed `ddgs` upstream; pin whichever version imports cleanly and keep the import in a try/except with the other name as fallback.

Full payloads stay in state; the trace stores only the first 200 characters.

---

## 5. Task 3A — single agent, autonomous tool use

- `ResearchAgent(tools=ALL_FIVE)` runs the loop: LLM sees the task, the tool list, and the compact observation log; returns one `AgentAction`; Python dispatches; observation appended; repeat until `finish` or the cap (8).
- Guards: schema retry once on malformed action; two identical consecutive calls → warning injected into the next prompt; hard cap.
- **Order is never coded.** The opening prompt only states the goal and available tools.
- **Replan (natural, not scripted):** the action schema has an optional `replan_reason`. When the LLM changes course because of an observation, the loop stamps a `replan{reason, from_plan, to_action}` event into state and trace. Realistic triggers: news returns few items → `web_search`; volatility is high → second shorter window; price call fails → retry with a different period. 
- **Demonstration, kept honest:** a clearly labelled **fault-injection switch** (`FAULT_INJECT={"get_news": "empty"}`) used in one notebook cell to prove the fallback and replan path. It is disclosed as a test harness, not hidden in the normal run.
- Output: `ResearchReport` (below).

---

## 6. Task 3B — two agents

### 6.1 Roles and enforced restriction
- **Agent A (Data Analyst):** `get_price_data`, `calculate_volatility`, `llm_sentiment`.
- **Agent B (Research Writer):** `web_search`, `get_news`.
- The dispatcher checks `tool in whitelist[agent]` **at runtime**; violation raises `CrossAgentToolAccessError`, which the loop catches and returns as a refused observation so the agent replans within its own tools. Prompts also state the limits, but the guard is what enforces them.
- Demo cell: B is forced to attempt `get_price_data` → refusal printed → B continues with `get_news`.
- Unit test: every cross-access pair is rejected.

### 6.2 Why the critique loop is genuine
v1 had Agent A "widen to web_search" and score headlines it never fetched, but A has neither `web_search` nor `get_news`. The corrected design uses that constraint:

- A can score sentiment (`llm_sentiment`) but cannot fetch headlines; B can fetch headlines (`get_news`) but cannot score them or compute volatility. So **neither agent can produce sentiment-scored news alone**.
- Flow:
  1. **A → brief v1:** price level, indicators, annualized volatility (90-day). No sentiment yet.
  2. **B** fetches headlines (`get_news`) and commentary (`web_search`), reviews the brief against a reporting checklist (needs scored sentiment and a short-window vol), and emits `CritiqueRequest{questions:[…], payload:{headlines:[…]}}` — structured output from the LLM, with a template fallback if the LLM call fails.
  3. **A** answers via its own tools: `llm_sentiment(headlines)` and `calculate_volatility(ticker, 30)` → `ClarificationResponse{answers:[{question_id, response, data}]}`.
  4. **B** incorporates the answers, referencing them by `question_id` in evidence, and writes the final report.
- Counter in state caps the loop at **one** cycle (spec: "one specific clarification request"). The loop is guaranteed by the system's structure, not by a faked trigger; the README states this plainly.

### 6.3 Typed handoff
`AgentBrief{ticker, as_of, price_level{current, week52_high, week52_low, ytd_pct}, volatility{annualized_90d, daily, band}, indicator_state{sma50_stance, sma200_stance, rsi, macd_hist, pct_b}, sentiment: Optional[...], analyst_notes}`.
B receives only the serialized brief. Round-trip test `AgentBrief.model_validate_json(brief.model_dump_json()) == brief`. The serialized JSON is printed in the notebook.

### 6.4 Graph
```
cache_check ──hit──▶ END (load + print "cache HIT")
     │miss
     ▼
agent_a_brief ─▶ agent_b_gather_and_critique ──needs_clarification & count<1──▶ agent_a_respond ─┐
                          │ no                                                                   │
                          └──────────────▶ agent_b_final_report ◀────────────────────────────────┘
                                                   ▼
                                          save_cache ─▶ END
```
`run_research(ticker)` compiles and invokes the graph. One call, no input.

### 6.5 Final report schema
```
ResearchReport{
  financial_health_summary{narrative, evidence_refs[]},
  risks[3]{title, detail, likelihood 1-5, evidence{source_tool, key_datum, url?}},
  hedge{strategy, instruments[], rationale, data_basis[], cost_note},
  meta{run_id, run_at, tool_call_count, replans, critique_cycles, memory_hits}}
```
Validators: exactly 3 risks, each with ≥ 1 evidence item; hedge `data_basis` ≥ 2 items. Make the hedge truly data-driven by computing in Python `expected_1sd_move_90d = price × annual_vol × √(90/252)` and requiring the hedge rationale to reference it (e.g., protective put/collar strike sized relative to that band). Add "not investment advice" text in the report.

---

## 7. Task 3C — memory and observability

- **Short-term:** `MemoryStore` keeps every `ToolResult`, the brief and the report for the session. A follow-up cell asks, e.g., "What sentiment class did the news analysis return, and how many headlines were scored?" The answer step receives the store contents and is told to answer **only** from it. Assertions: tool-call counter unchanged, trace shows `kind="memory", memory_reused=true`.
- **Persistent:** on completion save `cache/{TICKER}_{YYYY-MM-DD}.json` (date from `datetime.date.today()`, schema version inside). Next run checks it, prints `cache HIT — loaded brief for NVDA on <date>`, skips tool plan, and writes a `cache_load` trace line. Corrupt file → parse guard → invalidate → full run.
- **Trace:** `TraceLogger` wraps every tool call and LLM step. Line: `{ts_utc, run_id, kind(tool|llm|memory|cache), agent, tool, args, output (first 200 chars), duration_ms, ok, error}`. Append per call, flush immediately. `logs/agent_trace.jsonl` is committed and must contain a real full run plus the cache-hit run.
- Message trace: a printer renders each step in the notebook as `[Agent A] decide → tool(args) → observe → decide…`, handoffs as pretty JSON.

---

## 8. Bonus — Streamlit trace dashboard (optional, ~45 min)
`dashboards/trace_dashboard.py` (~60 lines): tool-call histogram per agent, duration bars, success/failure pie, run-ID filter. Screenshot into the README. Zero external accounts.

---

## 9. Failure handling
| Failure | Path |
|---|---|
| price data empty | `ok=false, hint` → agent retries other period or proceeds with partial data |
| news short | ladder → agent tries `web_search` (B) |
| web_search rate-limited | backoff once → skip, report notes "commentary unavailable" |
| LLM malformed action | one retry → skip step, log |
| `llm_sentiment` failure | retry → neutral sentinel with `estimate=true` |
| cross-agent tool use | `CrossAgentToolAccessError` → observation → replan |
| cache corrupt | invalidate, rerun |
| graph crash | wrapper returns partial JSON with `degraded=true` |

## 10. Testing
- Unit: each tool with mocked network; volatility exact on a known series; whitelist rejection; handoff round-trip; cache freshness/corruption; trace schema.
- Integration: end-to-end run with a stubbed LLM asserting `replans ≥ 1` under fault injection, `critique_cycles == 1`, no exceptions.
- Notebook "chaos" cell showing the fault-injection path.

## 11. Notebook plan (outputs visible)
1. Setup. 2. Each tool demo call (data types shown). 3. **3A:** single-agent run with trace and replan. 4. **3B:** whitelist-block demo; full two-agent run with message trace, brief JSON, critique request/response, final report. 5. **3C:** follow-up answered from memory (counter unchanged); second run → cache HIT. 6. Trace file preview + dashboard screenshot.

## 12. Milestones
| # | Work | Time |
|---|---|---|
| M1 | scaffold, tracing, LLM client (copy from Task 1) | 0.5 h |
| M2 | five tools + unit tests | 1 h |
| M3 | ReAct loop, single agent (3A), replan stamping | 1 h |
| M4 | schemas, whitelist, Agent A/B, graph, critique loop | 1.5 h |
| M5 | memory + cache + follow-up | 0.5 h |
| M6 | notebook run, README, CITATIONS | 1 h |
| M7 | dashboard (optional) | 0.5 h |

## 13. Risks
| Risk | Mitigation |
|---|---|
| DDG flakiness / package rename | backoff, try/except import, mocked tests, graceful degrade |
| LLM picks malformed actions | Pydantic action schema + retry + loop cap |
| LLM latency | per-step timeout; response cache for reruns |
| yfinance `.info` gaps | handled as None |

## 14. Interview talking points
Whitelist enforced at runtime versus prompted etiquette; tool-split design that makes the critique loop necessary; JSON-action protocol for provider independence; typed handoff round-trip; replan as an LLM-declared, logged event; fault injection disclosed rather than hidden; trace as the source of truth; hedge sized from computed 90-day 1σ move.

## 15. Changes from v1
- Fixed impossible flow (Agent A was asked to use `web_search`/news it doesn't own); critique loop now grounded in the tool split.
- Added explicit 3A single-agent stage (v1 skipped it).
- Replan is LLM-declared and logged; fault injection labelled as test harness.
- Dropped the shared core module; self-contained folder; simplified memory recall to an LLM answer from the store.
- Hedge made computable (1σ 90-day band); ddgs rename handled.

*AI-assisted plan; log it in `CITATIONS.md`.*
