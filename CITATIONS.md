# Citations — AI tool usage and adapted code

Citation format (required):

- AI-generated code / documents:
  `# AI-ASSISTED: <tool (model)>, Prompt: '<short prompt>', Date: <date>` (inline) — plus the per-file entries below.
- Adapted open-source code:
  `# SOURCE: <url>, file: <file>, Lines <a>-<b>` (inline, at the adapted location).

## AI assistants used

- **Cline (Claude Sonnet 5.5)** — AI coding assistant: implementation planning, code generation,
  tests, and documentation across all three projects.
- **GLM-5.3-Flash** (via an OpenAI-compatible gateway, `z-ai/glm-5.3-flash`) — runtime LLM for:
  news sentiment scoring, Buy/Hold/Sell signal reasoning, teacher-model dataset generation,
  LLM-as-judge scoring, and the agents' reasoning core.

## Per-file index

| File | Type | Entry |
|---|---|---|
| `docs/plan_task1_financial.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'revise the equity-research implementation plan against the requirements', Date: 2026-10-06` |
| `docs/plan_task2_genai.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'revise the fine-tuning pipeline plan (dataset, QLoRA config, evaluation)', Date: 2026-10-06` |
| `docs/plan_task3_agentic.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'revise the multi-agent system plan (3A/3B/3C stages, critique loop, memory, trace)', Date: 2026-10-06` |

*(Entries for source files are appended as the implementation proceeds — every AI-assisted
file carries the inline tag and a row here.)*

## Equity research assistant (`equity_research/`)

| File | Type | Entry |
|---|---|---|
| `equity_research/src/config.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'central config module: every constant with its reason', Date: 2026-10-06` |
| `equity_research/src/analysis/indicators.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'first-principles SMA/RSI/MACD/Bollinger with guards and tests-friendly purity', Date: 2026-10-06` |
| `equity_research/src/schemas.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'pydantic schemas with mechanical validators for LLM outputs', Date: 2026-10-06` |
| `equity_research/src/llm/prompts.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'all LLM prompts as documented constants with placeholders only', Date: 2026-10-06` |
| `equity_research/src/llm/llm_client.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'OpenAI-compatible LLM client: retries, JSON repair, cache, failure log', Date: 2026-10-06` |
| `equity_research/src/data/news.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'news retrieval ladder yfinance both shapes plus RSS fallbacks with dedupe', Date: 2026-10-06` |
| `equity_research/src/data/data_pipeline.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'market data pipeline with fetch ladder, cleaning, summary dictionary', Date: 2026-10-06` |
| `equity_research/src/analysis/analysis.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'analysis layer: per-headline sentiment with one call each and signal generation with fallback', Date: 2026-10-06` |
| `equity_research/src/reporting/report.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'HTML research brief with three-panel matplotlib chart embedded base64 and risk disclaimer', Date: 2026-10-06` |
| `equity_research/tests/*` (6 files + conftest) | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline pytest suite: indicator goldens/properties/slow-loop, pipeline, news, schemas, LLM client, hygiene', Date: 2026-10-06` |
| `equity_research/notebooks/equity_research.ipynb` | AI-assisted artifact | Generated and executed by Cline (Claude Sonnet 5.5); the LLM outputs inside were produced by GLM-5.3-Flash via the configured OpenAI-compatible gateway and replayed from the committed cache |
| `equity_research/README.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'project README: purpose, setup, provider switching, results summary', Date: 2026-10-06` |

## Supply-chain anomaly analyst (`supply_chain_finetuning/`)

| File | Type | Entry |
|---|---|---|
| `supply_chain_finetuning/src/config.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'central config: taxonomy, scenario matrix, gates, thresholds, paths', Date: 2026-10-06` |
| `supply_chain_finetuning/src/schemas.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'pydantic models: anomaly input, output contract with mechanical validators, teacher wrapper', Date: 2026-10-06` |
| `supply_chain_finetuning/src/prompts.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'all prompt text as documented constants: teacher, student baseline, judge', Date: 2026-10-06` |
| `supply_chain_finetuning/src/llm_client.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'shared LLM client: env-prefix settings, backoff, JSON parse, cache, usage CSV log', Date: 2026-10-06` |
| `supply_chain_finetuning/src/scenario_matrix.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'scenario matrix builder: 540 tuples plus stratified deterministic sampling', Date: 2026-10-06` |
| `supply_chain_finetuning/scripts/generate_dataset.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'teacher dataset generation with seven acceptance gates, resumable, usage-logged', Date: 2026-10-06` |
| `supply_chain_finetuning/scripts/diversity_report.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'dataset diversity report: token histograms, keyword coverage, heatmap, near-duplicates', Date: 2026-10-06` |
| `supply_chain_finetuning/scripts/build_jsonl.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'chat-format JSONL builder with exact stratified 120/15/15 split', Date: 2026-10-06` |
| `supply_chain_finetuning/evaluation/normalize.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'JSON extraction, schema validation and canonicalization for model outputs', Date: 2026-10-06` |
| `supply_chain_finetuning/evaluation/metrics.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'ROUGE-L, BERTScore, programmatic checks and ground guard for base vs tuned comparison', Date: 2026-10-06` |
| `supply_chain_finetuning/evaluation/judge.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'blinded LLM-as-judge with schema-validated scoring, shuffling and consistency gauge', Date: 2026-10-06` |
| `supply_chain_finetuning/evaluation/manual_audit.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'manual hallucination audit template generator and rate calculator (labels stay human)', Date: 2026-10-06` |
| `supply_chain_finetuning/tests/*` (7 files + conftest) | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline pytest suite: schemas, scenario matrix, generation gates, JSONL split, metrics, hygiene', Date: 2026-10-06` |
| `supply_chain_finetuning/notebooks/finetune_and_evaluate.ipynb` | AI-assisted artifact | Generated and validated by Cline (Claude Sonnet 5.5); to be executed by the user on Colab T4 |
| `supply_chain_finetuning/README.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'project README: purpose, pipeline overview, stage instructions, results placeholders', Date: 2026-10-06` |
| `supply_chain_finetuning/use_case.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'structured use-case statement: input/output contract, taxonomy, correctness definitions', Date: 2026-10-06` |
| `supply_chain_finetuning/TRAINING_NOTES.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'hyperparameter table with reasons, version-pinning notes, pre-committed loss responses, OOM playbook', Date: 2026-10-06` |
| `supply_chain_finetuning/data/teacher_system_prompt.txt` | Teacher-generated artifact | Verbatim copy of `src/prompts.py::TEACHER_SYSTEM`; dataset provenance record |

## Teacher-model prompts

- Not applicable to the equity research assistant (no synthetic training data).
- The fine-tuning project (when added) will store its full teacher system prompt
  verbatim in its data folder and notebook appendix for provenance auditing.

## Agentic financial research system (`agentic_research/`)

| File | Type | Entry |
|---|---|---|
| `agentic_research/src/config.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'env wiring, whitelists, caps with reasons, paths, fault-injection hook', Date: 2026-10-06` |
| `agentic_research/src/schemas.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'typed ReAct models: brief, critique, clarification, validated report', Date: 2026-10-06` |
| `agentic_research/src/tools/indicators.py` | Adapted code (copied from `equity_research/src/indicators.py` at commit `9b00d33`, `# SOURCE:` header) | Project-scoped reuse; per-file origin recorded inline |
| `agentic_research/src/runtime/llm_client.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'OpenAI-compatible client: retry, JSON repair, response cache, cache-hit flag', Date: 2026-10-06` |
| `agentic_research/src/runtime/tracing.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'JSONL trace logger with in-memory mirror and digest outputs', Date: 2026-10-06` |
| `agentic_research/src/runtime/memory.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'session memory store plus daily persistent cache with schema versioning and corruption guard', Date: 2026-10-06` |
| `agentic_research/src/prompts.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'all agent protocol, critique writer, clarification, composer and memory prompts as constants', Date: 2026-10-06` |
| `agentic_research/src/tools/*` (base, pricing, news, volatility, sentiment, websearch, sources, registry, __init__) | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'five tools with ToolResult contract, arg models, injected data sources, digest summaries', Date: 2026-10-06` |
| `agentic_research/src/agents/agents.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'ReAct agent loop: JSON-action protocol, duplicate and iteration guards, replan stamping, whitelist enforcement', Date: 2026-10-06` |
| `agentic_research/src/agents/graph.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'LangGraph state machine: cache check, two-agent roles, exactly-one critique cycle, save cache', Date: 2026-10-06` |
| `agentic_research/src/main.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'public entry points: single-agent mode, memory follow-up, CLI', Date: 2026-10-06` |
| `agentic_research/src/printing.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'human-readable trace renderer with per-agent summary counts', Date: 2026-10-06` |
| `agentic_research/tests/*` (10 files + conftest + helpers) | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline pytest suite: indicators goldens/edges, tools, agents, whitelist pairs, handoff, schemas, graph e2e with fault and crash injection, memory/cache, hygiene', Date: 2026-10-06` |
| `agentic_research/notebooks/agentic_research.ipynb` | AI-assisted artifact | Generated and executed by Cline (Claude Sonnet 5.5); agent reasoning produced by GLM-5.3-Flash via the configured OpenAI-compatible gateway; repeated calls replayed from the committed response cache |
| `agentic_research/dashboards/trace_dashboard.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'streamlit trace dashboard: run filter, per-agent tool histogram, duration bars, success pie, replan table', Date: 2026-10-06` |
| `agentic_research/README.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'project README: architecture diagram, guard semantics, run instructions, honest notes', Date: 2026-10-06` |
