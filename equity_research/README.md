# Equity Research Assistant

An automated equity research assistant that replicates the first-pass analytical work of a
junior analyst: it ingests real market data, computes technical indicators **from first
principles** (no TA-Lib, no indicator libraries), retrieves recent news from free sources, and
uses an LLM (any OpenAI-compatible provider) to produce **structured, Pydantic-validated**
news sentiment and a reasoned Buy/Hold/Sell recommendation. Everything lands in a one-page
HTML research brief with an embedded chart and a risk disclaimer.

## Project structure

```
equity_research/
  src/
    config.py            # constants, paths, thresholds
    schemas.py           # shared Pydantic contracts (summary, sentiment, signal)
    main.py              # CLI: full pipeline, artefacts, stale-cache fallback
    llm/                 # LLM boundary layer
      llm_client.py      # OpenAI-compatible client: retries, JSON repair, cache, failure log
      prompts.py         # every prompt as documented constants
    data/                # market + news ingestion
      data_pipeline.py   # OHLCV fetch ladder, cleaning, summary dictionary
      news.py            # news ladder (yfinance both shapes → Yahoo RSS → Google RSS)
    analysis/            # computation layer
      indicators.py      # SMA/RSI/MACD/Bollinger from first principles
      analysis.py        # per-headline sentiment + signal generation
    reporting/           # presentation layer
      report.py          # HTML brief with inline CSS + base64 chart
  tests/                 # offline pytest suite (50 tests)
  notebooks/             # executed demo notebook
  outputs/               # committed artefacts + LLM response cache
  logs/                  # runtime logs (gitignored)
```

## How it works

```
yfinance (3y daily OHLCV) ──▶ indicators (SMA50/200, RSI-14 Wilder, MACD 12-26-9, Bollinger 20-2σ)
        │                            │
news ladder (yfinance ▶ Yahoo RSS ▶ Google RSS)     summary dictionary (Pydantic)
        │                            │
        └──────────┬─────────────────┘
                   ▼
   GLM / any OpenAI-compatible LLM (temperature 0, JSON mode, Pydantic gates)
   ├── per-headline sentiment (one call per headline) ──▶ confidence-weighted aggregate
   └── recommendation over the evidence bundle ──▶ BUY / HOLD / SELL (3–5 sentences)
                   ▼
   outputs/report.html (single file: inline CSS + base64 chart + risk disclaimer)
```

## Setup

```bash
pip install -r requirements.txt

# LLM configuration (never commit the real file)
cp .env.example ../.env        # or keep .env next to this folder; dotenv walks up
```

`.env` variables:

```
LLM_BASE_URL=https://api.commandcode.ai/provider/v1
LLM_MODEL=z-ai/glm-5.3-flash
LLM_API_KEY=<your-key-here>
LLM_TIMEOUT_S=120
```

**Provider switching** — the client is provider-agnostic; change only these variables:

| Provider | LLM_BASE_URL | Example model |
|---|---|---|
| Default gateway | `https://api.commandcode.ai/provider/v1` | `z-ai/glm-5.3-flash` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | any routed model |

On Colab the notebook reads the key from Colab Secrets (`LLM_API_KEY`) instead of `.env`.

## Run

```bash
python -m src.main --ticker NVDA          # full pipeline
python -m src.main --ticker NVDA --no-cache   # bypass the LLM response cache
pytest -q                                  # offline verification suite (no network/LLM)
jupyter notebook notebooks/equity_research.ipynb
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<COLAB_NOTEBOOK_LINK_PLACEHOLDER>)

## Results (latest run, NVDA)

| Item | Value |
|---|---|
| Price history | 751 daily bars, 1095-day span (>= 2 years enforced) |
| Last price / daily / YTD | 238.90 / +2.12% / +28.10% |
| 52-week high / low | 240.10 / 164.27 (−0.50% from the high) · trailing P/E 30.20 |
| SMA 50 / SMA 200 | 218.92 / 201.02 (price above both; golden-cross structure) |
| RSI 14 / MACD histogram | 66.90 / +1.21 (fresh-cross: no) |
| Momentum vote | bullish (net +5 of 5 conditions) |
| News sentiment | 19 headlines, overall **+0.317 (positive)** — 9 positive / 3 negative / 7 neutral, 0 fallbacks |
| Recommendation | **BUY** (LLM-generated, validators passed: 3–5 sentences, multi-indicator reasoning) |

## Artifacts

| File | What it is |
|---|---|
| `outputs/summary.json` | Validated summary dictionary (price, 52w, P/E, YTD, indicator snapshot, momentum) |
| `outputs/sentiment.json` | Per-headline sentiment + confidence-weighted aggregate |
| `outputs/signal.json` | Recommendation with justification, drivers, risks |
| `outputs/chart.png` / `outputs/report.html` | 3-panel chart and the single-file HTML brief |
| `outputs/.llm_cache.json` | Committed response cache — re-runs replay identical LLM output at zero API spend |
| `logs/llm_failures.log` | Failure log (gitignored); the notebook displays the in-memory mirror |

## Engineering notes

- **Indicators are hand-verified**: golden values, edge cases (rising → RSI 100, flat → RSI 50),
  property tests, and an independent slow-loop re-implementation (`tests/test_indicators.py`).
- **Robustness**: fetch failure ladder (backoff → shorter window → friendly error + stale-cache
  fallback), null-bar cleaning, P/E unavailability rendered as "N/A (source unavailable)",
  news sources individually guarded, LLM repair-then-fallback flow — demonstrated live in the notebook.
- **Hygiene tests**: no date literals outside comments, prompt text lives only in `prompts.py`,
  no credential patterns anywhere in the project.
