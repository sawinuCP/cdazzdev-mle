# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'central config: loop caps, whitelists, thresholds, paths, fault-injection switch', Date: 2026-10-06
"""Central configuration — every tunable value lives here (no magic numbers)."""
from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_TRACE_JSONL = LOGS_DIR / "agent_trace.jsonl"
LLM_CACHE_JSON = OUTPUTS_DIR / ".llm_cache.json"
REPORT_JSON = OUTPUTS_DIR / "report.json"
REPORT_MD = OUTPUTS_DIR / "report.md"


def ensure_dirs() -> None:
    """Create output/log/cache directories on demand (idempotent)."""
    for directory in (OUTPUTS_DIR, LOGS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ── Market data ───────────────────────────────────────────────────────────────
DEFAULT_TICKER = "NVDA"
DEFAULT_PERIOD = "1y"
ALLOWED_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
WEEK52_BARS = 252
TRADING_DAYS_PER_YEAR = 252

# ── Indicators (same conventions as the equity research project) ─────────────
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
FRESH_CROSS_WINDOW = 3
BB_WINDOW = 20
BB_NUM_STD = 2.0
MIN_BARS_FOR_INDICATORS = 60  # tools tolerate shorter histories than the composer needs

# ── Volatility tool ───────────────────────────────────────────────────────────
VOL_WINDOW_DEFAULT = 90
VOL_WINDOW_SHORT = 30
VOL_WINDOW_MIN, VOL_WINDOW_MAX = 5, 252
VOL_BAND_LOW_MAX = 0.30        # annualized vol <= 0.30 -> "low"
VOL_BAND_HIGH_MIN = 0.60       # annualized vol >= 0.60 -> "high"; else "moderate"

# ── Hedge sizing ──────────────────────────────────────────────────────────────
HEDGE_HORIZON_DAYS = 90        # the 1-sigma band horizon stated in the objective

# ── News / sentiment ──────────────────────────────────────────────────────────
NEWS_DEFAULT_COUNT = 10
NEWS_MIN_COUNT, NEWS_MAX_COUNT = 1, 20
NEWS_PARTIAL_SHARE = 0.5       # partial coverage when >= half the requested headlines
SENTIMENT_MAX_HEADLINES = 10   # bounds the per-headline LLM calls
SENTIMENT_POSITIVE_THRESHOLD = 0.15
SENTIMENT_NEGATIVE_THRESHOLD = -0.15
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS_URL = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

# ── Web search ────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 6
WEB_SEARCH_SLEEP_S = 2.0       # politeness delay before each DDG request
WEB_SEARCH_TIMEOUT_S = 20.0
USER_AGENT = "agentic-financial-research/1.0 (market study)"

# ── LLM client ────────────────────────────────────────────────────────────────
ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"
ENV_TIMEOUT_S = "LLM_TIMEOUT_S"
DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
LLM_TEMPERATURE = 0.0
LLM_USE_RESPONSE_FORMAT = False  # json_object mode on this gateway burns the
                                 # whole budget on hidden reasoning and returns
                                 # empty content (finish=length); plain mode +
                                 # fence stripping + repair retries works
LLM_MAX_TOKENS = 2400          # thinking models burn reasoning tokens first; the
                               # composer's report JSON needs headroom (700 was
                               # observed empty even doubled to 1400)
LLM_TOKEN_BUDGET_MULTIPLIER = 2
LLM_MAX_ATTEMPTS = 3
LLM_BACKOFF_BASE_S = 1.5
LLM_REPAIR_ATTEMPTS = 1
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# ── Agent loop guards ─────────────────────────────────────────────────────────
SINGLE_AGENT_MAX_STEPS = 8
ROLE_MAX_STEPS = 6             # cap for each two-agent role
CRITIQUE_CYCLE_CAP = 1         # the critique loop runs exactly once
OBSERVATION_DIGEST_MAX_CHARS = 220
OBSERVATION_LOG_WINDOW = 6     # how many recent digests the LLM sees
ACTION_RETRY_ON_MALFORMED = 1

# ── Tool whitelists (runtime-enforced) ────────────────────────────────────────
TOOL_GET_PRICE_DATA = "get_price_data"
TOOL_GET_NEWS = "get_news"
TOOL_CALCULATE_VOLATILITY = "calculate_volatility"
TOOL_LLM_SENTIMENT = "llm_sentiment"
TOOL_WEB_SEARCH = "web_search"
ALL_TOOLS = (
    TOOL_GET_PRICE_DATA, TOOL_GET_NEWS, TOOL_CALCULATE_VOLATILITY,
    TOOL_LLM_SENTIMENT, TOOL_WEB_SEARCH,
)
WHITELISTS = {
    "single": frozenset(ALL_TOOLS),
    "agent_a": frozenset({TOOL_GET_PRICE_DATA, TOOL_CALCULATE_VOLATILITY, TOOL_LLM_SENTIMENT}),
    "agent_b": frozenset({TOOL_WEB_SEARCH, TOOL_GET_NEWS}),
}

# ── Fault injection (test harness ONLY — documented honestly in the README) ───
FAULT_INJECT: dict = {}        # e.g. {"get_news": "empty"} -> the tool returns a failed result

# ── Observability ─────────────────────────────────────────────────────────────
TRACE_OUTPUT_MAX_CHARS = 200
