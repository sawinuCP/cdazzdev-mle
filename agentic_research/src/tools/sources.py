# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'injectable data sources so tests can stub the network', Date: 2026-10-06
"""Injectable data sources.

Every tool goes through these four functions for network I/O; tests patch
THEM (``monkeypatch.setattr(tools.sources, "_fetch_history", ...)``) so the
suite runs fully offline. Attribute access at call time is deliberate.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import requests

from .. import config


def _fetch_history(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # lazy: tests stub the module

    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _news_raw(ticker: str) -> List[Dict[str, Any]]:
    import yfinance as yf

    return yf.Ticker(ticker).news or []


def _http_get(url: str) -> str:
    response = requests.get(url, timeout=config.WEB_SEARCH_TIMEOUT_S,
                            headers={"User-Agent": config.USER_AGENT})
    response.raise_for_status()
    return response.text


def _ddgs_text(query: str, max_results: int) -> List[Dict[str, Any]]:
    try:
        from ddgs import DDGS  # the renamed upstream package
    except ImportError:
        from duckduckgo_search import DDGS  # pinned fallback
    return list(DDGS().text(query, max_results=max_results))
