# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'central config module: every constant with its reason', Date: 2026-10-06
"""Central configuration for the equity research assistant.

Every tunable value lives here so that no magic numbers appear in the logic.
Comments capture *why* each value was chosen, so decisions are auditable.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # the equity_research folder
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
SUMMARY_JSON = OUTPUTS_DIR / "summary.json"
SENTIMENT_JSON = OUTPUTS_DIR / "sentiment.json"
SIGNAL_JSON = OUTPUTS_DIR / "signal.json"
REPORT_MD = OUTPUTS_DIR / "report.md"
REPORT_HTML = OUTPUTS_DIR / "report.html"
CHART_PNG = OUTPUTS_DIR / "chart.png"
LLM_CACHE_JSON = OUTPUTS_DIR / ".llm_cache.json"
LLM_FAILURE_LOG = LOGS_DIR / "llm_failures.log"


def ensure_dirs() -> None:
    """Create output/log directories on demand (idempotent)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Market data ───────────────────────────────────────────────────────────────
DEFAULT_TICKER = "NVDA"

# 3 years, not 2: the 200-day SMA alone consumes ~200 bars of warm-up, so a 2y fetch
# would leave under two years of *valid* indicator values. 3y guarantees >= 2 years
# of data AND of valid indicators.
LOOKBACK_PERIOD = "3y"
FALLBACK_PERIOD = "2y"  # shorter retry used only when the primary fetch fails
MIN_SPAN_DAYS = 730     # the spec floor: at least ~2 years of calendar coverage
FETCH_RETRIES = 3
FETCH_BACKOFF_BASE_S = 1.5
TRADING_DAYS_PER_YEAR = 252
WEEK52_BARS = 252  # last 252 bars of High/Low define the 52-week range

# ── Indicators ────────────────────────────────────────────────────────────────
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
MACD_HIST_TREND_WINDOW = 5  # look-back used to label the histogram trend
FRESH_CROSS_WINDOW = 3      # a sign change within the last N bars counts as "fresh"
BB_WINDOW = 20
BB_NUM_STD = 2.0
CHART_WINDOW_BARS = 260  # bars shown in the report chart (recent context, all indicators valid)

# ── Momentum vote (simple condition counting; points are +/-1 each) ──────────
MOMENTUM_BULLISH_NET_THRESHOLD = 2   # net >= +2  -> bullish
MOMENTUM_BEARISH_NET_THRESHOLD = -2  # net <= -2  -> bearish
RSI_BULLISH_BAND = (50.0, 70.0)      # constructive but not stretched
RSI_BEARISH_BAND = (30.0, 50.0)
RSI_OVERBOUGHT = 70.0                # recorded as a note, never as a vote
RSI_OVERSOLD = 30.0

# ── News retrieval ladder ─────────────────────────────────────────────────────
MIN_HEADLINES = 10
MAX_HEADLINES = 20
PARTIAL_COVERAGE_MIN = 5  # >=5 but <10 -> "partial"; fewer -> "low"
HTTP_TIMEOUT_S = 15
USER_AGENT = "equity-research-assistant/1.0 (market study)"
YAHOO_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)
GOOGLE_RSS_URL = (
    "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
)

# ── LLM client ────────────────────────────────────────────────────────────────
ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"
ENV_TIMEOUT_S = "LLM_TIMEOUT_S"
# Defaults mirror .env.example; the client is provider-agnostic (Groq, OpenRouter
# and other OpenAI-compatible gateways work by changing environment variables only).
DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_TIMEOUT_S = 120.0
LLM_TEMPERATURE = 0.0            # analyst output must be reproducible
LLM_MIN_MAX_TOKENS = 700         # thinking-style models spend budget on reasoning tokens
LLM_TOKEN_BUDGET_MULTIPLIER = 2  # one retry with doubled budget when content is empty
LLM_MAX_ATTEMPTS = 3             # exponential backoff on transport / 429 / 5xx errors
LLM_BACKOFF_BASE_S = 1.5
LLM_REPAIR_ATTEMPTS = 1          # one extra call that appends the exact validation error
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# ── Sentiment aggregation ─────────────────────────────────────────────────────
SENTIMENT_POSITIVE_THRESHOLD = 0.15
SENTIMENT_NEGATIVE_THRESHOLD = -0.15
TOP_HEADLINES_COUNT = 3

# ── Signal validation ─────────────────────────────────────────────────────────
SENTENCE_COUNT_MIN = 3
SENTENCE_COUNT_MAX = 5
# Splits only at sentence boundaries followed by a capital letter, so decimals
# ("1.5") and abbreviations ("U.S.") do not inflate the count.
SENTENCE_SPLIT_REGEX = r"(?<=[.!?])\s+(?=[A-Z])"
MIN_INDICATOR_MENTIONS = 2  # mechanical proxy for "reasoning over combinations"
INDICATOR_CONCEPTS = (
    "rsi",
    "macd",
    "sma",
    "moving average",
    "bollinger",
    "sentiment",
    "momentum",
    "volume",
)
