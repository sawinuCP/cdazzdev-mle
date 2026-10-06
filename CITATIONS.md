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
| `equity_research/src/indicators.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'first-principles SMA/RSI/MACD/Bollinger with guards and tests-friendly purity', Date: 2026-10-06` |
| `equity_research/src/schemas.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'pydantic schemas with mechanical validators for LLM outputs', Date: 2026-10-06` |
| `equity_research/src/prompts.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'all LLM prompts as documented constants with placeholders only', Date: 2026-10-06` |
| `equity_research/src/llm_client.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'OpenAI-compatible LLM client: retries, JSON repair, cache, failure log', Date: 2026-10-06` |
| `equity_research/src/news.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'news retrieval ladder yfinance both shapes plus RSS fallbacks with dedupe', Date: 2026-10-06` |
| `equity_research/src/data_pipeline.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'market data pipeline with fetch ladder, cleaning, summary dictionary', Date: 2026-10-06` |
| `equity_research/src/analysis.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'analysis layer: per-headline sentiment with one call each and signal generation with fallback', Date: 2026-10-06` |
| `equity_research/src/report.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'HTML research brief with three-panel matplotlib chart embedded base64 and risk disclaimer', Date: 2026-10-06` |
| `equity_research/src/main.py` | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'CLI entry point: full pipeline with friendly failures and stale-cache fallback', Date: 2026-10-06` |
| `equity_research/tests/*` (6 files + conftest) | AI-assisted code | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline pytest suite: indicator goldens/properties/slow-loop, pipeline, news, schemas, LLM client, hygiene', Date: 2026-10-06` |
| `equity_research/notebooks/equity_research.ipynb` | AI-assisted artifact | Generated and executed by Cline (Claude Sonnet 5.5); the LLM outputs inside were produced by GLM-5.3-Flash via the configured OpenAI-compatible gateway and replayed from the committed cache |
| `equity_research/README.md` | AI-assisted document | `# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'project README: purpose, setup, provider switching, results summary', Date: 2026-10-06` |

*(Teacher-model prompts do not apply to this project — it generates no synthetic training data.)*

## Teacher-model prompts

- Not applicable to the equity research assistant (no synthetic training data).
- The fine-tuning project (when added) will store its full teacher system prompt
  verbatim in its data folder and notebook appendix for provenance auditing.
