# Citations — AI tool usage and adapted code

AI tools were used during development, as permitted by the submission policy.
This file discloses where, per the citation requirement.

## Tooling

- **Cline (Claude)** — coding assistant used for drafting and refactoring code,
  tests and documentation across the three projects.
- **GLM-5.3-Flash** (OpenAI-compatible gateway) — runtime LLM in all three
  systems: news-sentiment scoring, the B/H/S signal, the supply-chain teacher
  model, the LLM-as-judge, and the agents' reasoning core.
- **yfinance / duckduckgo-search / Google & Yahoo RSS** — public data sources.

## Where assistance was used

| Area | Files | Nature |
|---|---|---|
| Shared LLM client (retry / JSON repair / response cache) | `equity_research/src/llm/llm_client.py`, `supply_chain_finetuning/src/llm_client.py`, `agentic_research/src/runtime/llm_client.py` | generated, then human-reviewed and hardened against real gateway behaviour |
| Prompt definitions (all prompts as constants) | `*/src/**/prompts.py`, `supply_chain_finetuning/data/teacher_system_prompt.txt` | generated; wording iterated against the live models |
| Pydantic contracts & validators | `*/src/**/schemas.py` | generated; validator rules derived from the domain requirements |
| Market/news data layer | `equity_research/src/data/*`, `equity_research/src/analysis/indicators.py` | generated; indicator maths verified against golden values and a slow-loop re-implementation |
| Agent runtime (models, tools, loops, graph, memory, trace) | `agentic_research/src/**` | generated; whitelist, guard and cache behaviour proven by the offline test suite |
| Fine-tuning pipeline | `supply_chain_finetuning/src/*`, `scripts/*`, `evaluation/*` | generated; gating and metrics validated offline, training runs on Colab |
| Tests, notebooks, dashboards, READMEs | `*/tests/*`, `*/notebooks/*`, `agentic_research/dashboards/*`, `**/README.md` | generated and executed; notebooks contain outputs from live runs replayed through the committed response caches |
| Adapted code | `agentic_research/src/tools/indicators.py` | adapted from `equity_research/src/analysis/indicators.py` (own Task-1 file, copied to keep the folder self-contained) |

Notes:

- The supply-chain teacher system prompt is stored verbatim in
  `supply_chain_finetuning/data/teacher_system_prompt.txt` as dataset
  provenance.
- Implementation-planning documents were used during development and removed
  from the repository afterwards.
