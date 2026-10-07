"""The data-access layer for the agents.

Re-exports the five tools, the dispatcher and the digester helpers so callers
can keep the stable interface ``from src import tools`` /
``tools.dispatch(...)``. The per-tool implementations live in single-purpose
modules; network I/O is funnelled through ``sources`` (test injection point).
"""
from __future__ import annotations

from . import base, sources            # noqa: F401  (sources = mock injection point)
from .base import _finite, _now_iso    # noqa: F401
from .base import make_result          # noqa: F401
from .news import (                    # noqa: F401
    GetNewsArgs, get_news, news_digest, _normalize_pubdate, _normalize_title_key,
    _normalize_yf_item, _parse_rss,
)
from .pricing import (                 # noqa: F401
    GetPriceDataArgs, get_price_data, price_digest,
)
from .registry import (                # noqa: F401
    allowed_tools_block, digest_for, dispatch, tool_signature,
)
from .sentiment import (               # noqa: F401
    LlmSentimentArgs, llm_sentiment, sentiment_digest,
)
from .volatility import (              # noqa: F401
    CalculateVolatilityArgs, calculate_volatility, volatility_digest,
)
from .websearch import (               # noqa: F401
    WebSearchArgs, web_search, search_digest,
)
