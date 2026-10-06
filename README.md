# CDAZZDEV-MLE — Machine Learning Engineering Projects

Self-contained machine-learning engineering projects:

| Folder | Project | What it does |
|---|---|---|
| `equity_research/` | **Equity Research Assistant** | Ingests real market data (yfinance), computes technical indicators from first principles, then uses an LLM to produce structured news sentiment and a reasoned Buy/Hold/Sell signal with validated structured output. Renders a one-page HTML research brief. |

## Quickstart

```bash
git clone https://github.com/<account>/<repo>.git
cd <repo>
pip install -r requirements.txt

# LLM access (any OpenAI-compatible provider — the default gateway, Groq,
# or OpenRouter work by changing environment variables only)
cp equity_research/.env.example .env    # then paste your key into .env (never commit it)
```

Run the project:

```bash
cd equity_research
python -m src.main --ticker NVDA
pytest -q                      # offline verification suite (no network / no LLM)
jupyter notebook notebooks/equity_research.ipynb
```

## Notes for reviewers

- The notebook is committed **with executed outputs** (visible results in every cell).
- The LLM response cache (`equity_research/outputs/.llm_cache.json`) is committed so the
  notebook re-runs to **identical output with zero API spend**.
- No credentials exist anywhere in this repository; the LLM client is provider-agnostic and
  configured purely through environment variables (see `equity_research/.env.example`).
- `CITATIONS.md` documents all AI tool usage; `REFLECTION.md` summarises decisions and limitations.
