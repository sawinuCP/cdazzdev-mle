# Task 1 — Financial AI: LLM-Powered Equity Research Assistant (Revised Plan v2)

> Repo folder: `task1_financial/` · Max score 100 (+5 report bonus, +5 video) · Est. effort 4–5 h
> Prepared: 2026-10-06 · Supersedes `plan_equity_research.md`

---

## 0. Repo-level rules (apply to all three tasks)

The assessment fixes the submission shape. Plans v1 used different folder names and omitted two mandatory files.

```
CDAZZDEV-MLE-<YourName>/            <- PUBLIC repo, exact naming
├── README.md                       # index + how to run each task
├── CITATIONS.md                    # mandatory, root level
├── REFLECTION.md                   # mandatory, <= 600 words, covers all tasks
├── .gitignore  .env.example  requirements.txt
├── task1_financial/
├── task2_genai/
└── task3_agentic/
```

- Each task folder is **self-contained** (own `src/`, own LLM client). No shared `common/` package. Colab then needs only `git clone` + `pip install`.
- Each task README carries a **Colab badge**. Notebooks are committed **with executed outputs**.
- **No keys anywhere.** Colab Secrets (`google.colab.userdata`) or `getpass`; `.env` is gitignored.
- AI usage is cited inline using the exact required format:
  `# AI-ASSISTED: Claude (Sonnet 5.5), Prompt: '<short prompt>', Date: 2026-10-06`
  plus one line per file in `CITATIONS.md`. Adapted open-source code uses `# SOURCE: <url>, file: <f>, Lines a-b`.
- Pre-send checklist (the assessment's list): incognito check on repo, outputs visible, grep for keys, CITATIONS + REFLECTION present.

---

## 1. Objective

Build an automated equity research assistant that (A) ingests real market data and computes indicators from first principles, then (B) uses an LLM to score news sentiment per headline and to produce a reasoned Buy/Hold/Sell, with validated structured output. Bonus: one-page HTML brief.

**Success criteria**
- ≥ 2 years of daily OHLCV, **no date literals** in logic.
- SMA50, SMA200, RSI14 (Wilder), MACD(12,26,9), Bollinger(20,2σ) without TA-Lib, backed by tests.
- ≥ 10 headlines from a free source.
- Every LLM output is Pydantic-validated; failures are caught, logged, degraded.
- Notebook runs top to bottom with visible outputs.

**Design thesis:** the weight sits on indicator correctness (25) and reasoning quality (15). So: verifiable math first, then an LLM that reasons over validated evidence and never calculates.

---

## 2. Rubric traceability

| Criterion (pts) | Implemented by | Evidence |
|---|---|---|
| OHLCV fetch, ≥ 2y, no hardcoded dates (10) | `data_pipeline.py` | notebook cell printing first/last date + span |
| Five indicators correct (25) | `indicators.py` | `tests/test_indicators.py` green |
| ≥ 10 headlines (10) | `news.py` ladder | notebook table of headlines |
| Summary dict complete (10) | `build_summary()` + `SummaryStats` | printed dict / `outputs/summary.json` |
| Robustness, no magic numbers, readable (5) | `config.py`, NaN guards | robustness tests |
| Per-headline JSON + aggregation (10) | `analysis.py::score_headlines` | `outputs/sentiment.json` |
| Signal reasons over combinations (15) | `analysis.py::generate_signal` | `outputs/signal.json` |
| Validation, failures logged (10) | `schemas.py`, `llm_client.py` | repair demo cell + `logs/llm_failures.log` |
| Prompt separation (5) | `prompts.py` | no prompt text elsewhere (grep test) |
| Bonus report (+5) | `report.py` | `outputs/report.html` |

---

## 3. Layout

```
task1_financial/
├── README.md                # Colab badge, results summary, how to run
├── notebook/task1_financial.ipynb
├── src/
│   ├── config.py            # ALL constants: ticker, period, windows, thresholds, env names
│   ├── indicators.py        # SMA / RSI / MACD / Bollinger
│   ├── data_pipeline.py     # fetch, clean, summary
│   ├── news.py              # headline ladder + dedupe
│   ├── schemas.py           # Pydantic models
│   ├── prompts.py           # every prompt as a constant
│   ├── llm_client.py        # OpenAI-compatible client, retry, JSON parse, cache
│   ├── analysis.py          # sentiment + signal pipelines
│   ├── report.py            # HTML + matplotlib chart
│   └── main.py              # python -m src.main --ticker NVDA
├── tests/
└── outputs/                 # summary.json sentiment.json signal.json report.html chart.png .llm_cache.json
```

---

## 4. Task 1A — Data pipeline

### 4.1 OHLCV
- `yf.Ticker(t).history(period=LOOKBACK_PERIOD, interval="1d", auto_adjust=False)` with `LOOKBACK_PERIOD = "3y"`.
- **Why 3y, not 2y:** SMA200 consumes ~200 bars of warm-up, so a 2y fetch leaves under two years of valid indicator values. 3y keeps ≥ 2y of data *and* of valid indicators. After fetching, assert `(index[-1] - index[0]).days >= 730` and print the span.
- `auto_adjust=False`: unadjusted closes match textbook formulas; documented in a code comment.
- Cleaning: drop rows with null OHLC, log count dropped, keep a `data_quality` note.
- Failure ladder: retry with backoff → shorter period (`"2y"`) → `MarketDataError` → caller uses stale cached summary flagged `stale=True`. No bare tracebacks.
- Date-literal test: grep `src/*.py` for `\d{4}-\d{2}-\d{2}` **excluding comment lines** (citation comments contain dates).

### 4.2 Indicators from first principles
```python
sma(n)   = close.rolling(n, min_periods=n).mean()

delta    = close.diff()
gain     = delta.clip(lower=0);  loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
rs       = avg_gain / avg_loss        # avg_loss==0 -> RSI 100 ; both 0 -> 50
rsi      = 100 - 100 / (1 + rs)

macd     = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
signal   = macd.ewm(span=9, adjust=False).mean()      # EMA of the MACD series
hist     = macd - signal

mid      = close.rolling(20).mean();  sd = close.rolling(20).std(ddof=0)
upper, lower = mid + 2*sd, mid - 2*sd
pct_b    = (close - lower) / (upper - lower)
```
Reasoning to state in comments:
- Wilder's smoothing is `alpha = 1/n`, not `span = n`. `span=n` would use a different decay and shift RSI levels.
- `adjust=False` seeds from the first value rather than an SMA seed; the difference vanishes after warm-up. Document it.
- Population σ (`ddof=0`) for Bollinger is the conventional charting choice.
- Pandas `rolling/ewm` is still "from first principles": no indicator library is used.

### 4.3 Tests (protects the 25 pts)
1. **Golden:** a short hand-computed series (e.g., 30 closes) with `pytest.approx`.
2. **Edge cases:** strictly rising series → RSI → 100; flat series → no crash, RSI = 50.
3. **Properties:** RSI ∈ [0,100]; upper ≥ mid ≥ lower; SMA(n) at t = mean of last n closes; `hist == macd - signal`.
4. **Independent re-implementation:** a slow explicit loop for RSI/SMA inside the test file, compared to the vectorized version.
5. Optional: compare 2–3 dates against a public charting site and note it in the notebook.

### 4.4 News ladder (≥ 10 headlines)
1. `Ticker(t).news`. Handle **both** response shapes: older flat (`title`, `publisher`, `link`) and newer nested under `content` (`content.title`, `content.provider.displayName`, `content.canonicalUrl.url`). yfinance changed this; coding for both avoids a silent zero.
2. If < 10: Yahoo Finance RSS (`requests` + `xml.etree`, timeout 15 s).
3. If still < 10: Google News RSS search.
Dedupe on normalized title, cap at 20, keep source + date. Record `news_coverage ∈ {full, partial, low}`.

### 4.5 Summary dictionary
`current_price, prev_close, daily_return_pct, week52_high, week52_low, pe_trailing (None → "N/A"), ytd_return_pct, indicators_snapshot, momentum_signal, momentum_components`.
- 52-week high/low: last 252 bars of `High`/`Low` (derived from the data, not a constant date).
- **YTD:** last close of the *previous calendar year* → latest close (fall back to first close of the year if the history lacks it). v1 used the first close of the current year, which understates YTD by one day's move.
- P/E: `info.get("trailingPE")` wrapped in try/except.

**Momentum signal (kept simple):** count conditions with constants in `config.py`:
`price > SMA50`, `price > SMA200`, `SMA50 > SMA200`, `MACD hist > 0`, `RSI ∈ (50,70)` = bullish points; opposites = bearish points; RSI > 70 or < 30 recorded as a note, not a vote. Net ≥ +2 → bullish, ≤ −2 → bearish, else neutral. Pass the components dict to the LLM.

### 4.6 Robustness
| Failure | Handling |
|---|---|
| yfinance empty / outage | backoff → shorter period → stale cache |
| null OHLC rows | drop + log + note |
| news < 10 after ladder | proceed, `news_coverage="low"` |
| `.info` fails | P/E = None |
| indicator warm-up NaN | min-history guard; use last valid value |

---

## 5. Task 1B — LLM layer

### 5.1 Client (`llm_client.py`)
- OpenAI-compatible SDK. Config from env: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Default model **GLM-5.3-Flash** via your verified gateway; **Groq (`llama-3.3-70b-versatile`) or OpenRouter are drop-in alternatives** by changing three env vars. State this in the README so a reviewer with a Groq key can re-run.
- Temperature 0. `max_tokens ≥ 700` and **one retry with doubled budget on empty content** (thinking-model behavior you verified).
- Exponential backoff on 429/5xx/timeouts.
- JSON path: request `response_format={"type":"json_object"}`; strip code fences; on parse/validation failure do **one** retry that appends the exact validation error.
- **Response cache** keyed by `sha256(model + prompt)` in `outputs/.llm_cache.json`, committed. A reviewer re-running the notebook gets identical output with zero API spend.
- Every failure is appended to `logs/llm_failures.log` (timestamp, stage, error).

### 5.2 Per-headline sentiment
- **One call per headline**, matching the spec literally ("pass each headline"). 10–20 calls are cheap and the cache makes re-runs free. v1 batched 5 per call, which risks a "did not pass each headline" reading and needs extra repair logic.
- `HeadlineSentiment{headline, sentiment: Literal["positive","negative","neutral"], confidence: float (0–1), brief_reason}`.
- Aggregate: `score = Σ(sign(s)·confidence) / Σ(confidence)` ∈ [−1, 1]; neutral has sign 0 but still counts in the denominator (so a neutral-heavy feed pulls toward 0). Label: > 0.15 positive, < −0.15 negative, else neutral (constants). Also output class counts and top 3 headlines by confidence.
- Invalid item after retry → neutral sentinel with `confidence=0`, flagged `fallback=True`, logged.

### 5.3 Signal reasoning
- Input is an **evidence bundle**, not raw arrays: indicator snapshot, momentum components, 5-day MACD-hist trend, YTD, distance from 52w high, sentiment score + top headlines.
- System prompt contract: act as an equity analyst; justify in 3–5 sentences by *relating* indicators to each other (e.g., price above both MAs but MACD histogram shrinking and RSI > 70 = stretched trend); no restating isolated values; no numbers outside the bundle.
- `TechnicalSignal{action: Literal["BUY","HOLD","SELL"], justification, key_drivers: list[str], risk_notes}`.
- Mechanical validators (graded properties enforced in code):
  - sentence count 3–5 using `re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)` so decimals and "U.S." don't inflate the count;
  - justification mentions ≥ 2 distinct indicator names (RSI, MACD, SMA, Bollinger, sentiment…), a cheap proxy for "combination reasoning".
- On failure: one retry with exact feedback → else rules-based fallback (`generated_by="rules_fallback"`), logged.

### 5.4 Prompt separation
All prompts live in `prompts.py` as constants/templates with placeholders. A test asserts no multi-line string literals containing "You are" exist outside it, and that every template formats against the real payload.

---

## 6. Bonus — report
`report.py` builds Markdown then HTML (inline CSS, no JS): snapshot table → technical outlook with a 3-panel matplotlib chart (price + SMAs + Bollinger / RSI / MACD) embedded as base64 → sentiment summary with top 3 headlines → LLM recommendation + reasoning → **risk disclaimer**. One portable file. README explains browser print-to-PDF.

---

## 7. Notebook plan (`task1_financial.ipynb`, outputs visible)
1. Setup: clone repo, `pip install -r requirements.txt`, keys via Colab Secrets.
2. Fetch data → print span, tail.
3. Indicators → table of last rows + one chart.
4. Tests cell: `!pytest -q task1_financial/tests` with visible green output.
5. News → table of ≥ 10 headlines.
6. Summary dict → pretty print.
7. Sentiment → JSON per headline + aggregate.
8. Signal → justification + validator results; **one deliberate failure-injection cell** showing a bad LLM output being caught, repaired, logged.
9. Report render + display.

---

## 8. Milestones
| # | Work | Time |
|---|---|---|
| M1 | scaffold, config, LLM client smoke test | 0.5 h |
| M2 | indicators + tests | 1 h |
| M3 | data pipeline, news ladder, summary | 1 h |
| M4 | sentiment + signal + validation | 1 h |
| M5 | report + notebook run in Colab + README/CITATIONS | 1 h |

---

## 9. Risks
| Risk | Mitigation |
|---|---|
| yfinance rate limit / shape change | ladder, both news shapes, stale-cache fallback, pin version |
| LLM gateway not reproducible for reviewer | env-switchable provider + committed cache |
| Empty LLM content | token floor + doubled retry |
| Time overrun | skip bonus report first |

---

## 10. Interview talking points
Unadjusted closes; Wilder `alpha=1/14`; population σ; why 3y not 2y; per-headline calls vs batching; confidence-weighted aggregation and its neutral handling; validators that enforce graded properties mechanically; fallback-first design; cache for zero-spend reproducibility; "LLM reasons over evidence, never calculates".

## 11. Changes from v1
- Folder/file names aligned to assessment; REFLECTION.md added; self-contained task folder.
- 3y fetch to guarantee ≥ 2y of valid indicators; YTD definition fixed.
- Per-headline calls instead of batches; simpler momentum rule; combination-check validator.
- News parsing handles both yfinance shapes; date-literal test excludes citation comments.
- Provider-agnostic client documented for reviewer reproducibility.

*AI-assisted plan; log it in `CITATIONS.md`.*
