# CDAZZDEV-MLE — Machine Learning Engineering Projects

Self-contained machine-learning engineering projects:

| Folder | Project | What it does |
|---|---|---|
| `equity_research/` | **Equity Research Assistant** | Ingests real market data (yfinance), computes technical indicators from first principles, then uses an LLM to produce structured news sentiment and a reasoned Buy/Hold/Sell signal with validated structured output. Renders a one-page HTML research brief. |
| `supply_chain_finetuning/` | **Supply-Chain Anomaly Analyst (QLoRA)** | Teacher-model pipeline generates a 12-family stratified fine-tuning dataset from one real incident brief; QLoRA fine-tunes Phi-3-mini-4k-instruct on Colab T4; evaluated base-vs-tuned via programmatic metrics, a blinded LLM-judge and a manual hallucination audit. |
| `agentic_research/` | **Agentic Financial Research (two agents)** | Two runtime-whitelisted agents research a ticker through a LangGraph state machine: the quant agent builds a typed brief, the news agent gathers headlines and critiques the gaps, the quant agent answers with its own tools, and the news agent writes the validated three-risk report. All numbers are computed in Python; session memory answers follow-ups without tools; the day-persisted cache makes the second run tool-free. |

## Quickstart

```bash
git clone https://github.com/<account>/<repo>.git
cd <repo>
pip install -r requirements.txt

# LLM access (any OpenAI-compatible provider — the default gateway, Groq,
# or OpenRouter work by changing environment variables only)
cp agentic_research/.env.example .env    # then paste your key into .env (never commit it)
```

Run the projects:

```bash
# project 1: equity research assistant
cd equity_research
python -m src.main --ticker NVDA
pytest -q                      # offline verification suite (no network / no LLM)
jupyter notebook notebooks/equity_research.ipynb

# project 2: QLoRA supply-chain analyst (dataset is generated; training runs on Colab T4)
cd supply_chain_finetuning
pytest -q                      # offline pipeline/evaluation suite
# then open notebooks/finetune_and_evaluate.ipynb (see its README section)

# project 3: agentic research system
cd agentic_research
python -m pytest tests -q      # fully offline suite (mocked LLM + data sources)
python -m src.main --ticker NVDA            # two-agent research run
python -m src.main --ticker NVDA --mode single   # single agent, all five tools
python -m src.main --ticker NVDA            # same day again = cache HIT (no tool calls)
```

## Notes for reviewers

- The notebook is committed **with executed outputs** (visible results in every cell).
- The LLM response cache (`equity_research/outputs/.llm_cache.json`) is committed so the
  notebook re-runs to **identical output with zero API spend**.
- No credentials exist anywhere in this repository; the LLM client is provider-agnostic and
  configured purely through environment variables (see `equity_research/.env.example`).
- `CITATIONS.md` documents all AI tool usage; `REFLECTION.md` summarises decisions and limitations.
