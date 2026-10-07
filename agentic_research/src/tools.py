# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'five tools with ToolResult contract, arg models, injected data sources, digest summaries', Date: 2026-10-06
"""The five research tools.

Contract: every tool returns a ``ToolResult`` and NEVER raises - bad arguments
or failures become ok=False with a useful ``hint``. Data sources are module
level functions (``_fetch_history``, ``_http_get``, ``_ddgs_text``) so tests can
inject mocks without network access. Full payloads stay in agent state; the
trace only records a digest.
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from . import config, indicators
from .llm_client import LLMClient
from .schemas import HeadlineSentiment, ToolResult
from .prompts import SENTIMENT_SYSTEM, SENTIMENT_USER

_LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> Optional[float]:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if (isinstance(as_float, float) and math.isnan(as_float)) else as_float


def make_result(ok: bool, data: Any = None, error: Optional[str] = None,
                hint: Optional[str] = None, source: str = "") -> ToolResult:
    """Single construction point for every ToolResult."""
    return ToolResult(ok=ok, data=data, error=error, hint=hint,
                      source=source, fetched_at=_now_iso())


# ── Injectable data sources (tests monkeypatch these) ─────────────────────────
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


# ── Argument models (bad args -> ok=False, never an exception) ────────────────
class GetPriceDataArgs(BaseModel):
    ticker: str = Field(min_length=1)
    period: str = config.DEFAULT_PERIOD


class GetNewsArgs(BaseModel):
    ticker: str = Field(min_length=1)
    n: int = Field(default=config.NEWS_DEFAULT_COUNT, ge=config.NEWS_MIN_COUNT,
                   le=config.NEWS_MAX_COUNT)


class CalculateVolatilityArgs(BaseModel):
    ticker: str = Field(min_length=1)
    window: int = Field(default=config.VOL_WINDOW_DEFAULT, ge=config.VOL_WINDOW_MIN,
                        le=config.VOL_WINDOW_MAX)


class LlmSentimentArgs(BaseModel):
    headlines: List[str] = Field(min_length=1)


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=3)


ARG_MODELS: Dict[str, type] = {
    config.TOOL_GET_PRICE_DATA: GetPriceDataArgs,
    config.TOOL_GET_NEWS: GetNewsArgs,
    config.TOOL_CALCULATE_VOLATILITY: CalculateVolatilityArgs,
    config.TOOL_LLM_SENTIMENT: LlmSentimentArgs,
    config.TOOL_WEB_SEARCH: WebSearchArgs,
}

# ── Tool 1: get_price_data ────────────────────────────────────────────────────
def get_price_data(ticker: str, period: str) -> ToolResult:
    """Daily OHLCV tail plus an indicator snapshot (built in Python)."""
    try:
        args = GetPriceDataArgs(ticker=ticker, period=period)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"period must be one of {', '.join(config.ALLOWED_PERIODS)}",
                           source="yfinance")
    if args.period not in config.ALLOWED_PERIODS:
        return make_result(False, error=f"period {args.period!r} not allowed",
                           hint=f"period must be one of {', '.join(config.ALLOWED_PERIODS)}",
                           source="yfinance")
    try:
        frame = _fetch_history(args.ticker, args.period)
    except Exception as exc:  # noqa: BLE001 - converted into a handled outcome
        return make_result(False, error=f"{type(exc).__name__}: {exc}",
                           hint="try a shorter period or web_search", source="yfinance")
    if frame.empty:
        return make_result(False, error="empty price frame",
                           hint="try a shorter period or web_search", source="yfinance")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().dropna(subset=["Open", "High", "Low", "Close"])
    close = frame["Close"].astype(float)
    price = _finite(close.iloc[-1])
    prev_close = _finite(close.iloc[-2]) if len(close) >= 2 else None

    indicator_state: Dict[str, Any] = {"sma50_stance": "unknown", "sma200_stance": "unknown",
                                       "rsi": None, "macd_hist": None,
                                       "macd_fresh_cross": False, "pct_b": None}
    try:
        ind = indicators.compute_all(close)
        last = ind.iloc[-1]
        sma50, sma200 = _finite(last.get("sma_50")), _finite(last.get("sma_200"))

        def _stance(value: Optional[float], reference: Optional[float]) -> str:
            return "unknown" if value is None or reference is None else \
                ("above" if value > reference else "below")

        indicator_state = {
            "sma50_stance": _stance(price, sma50),
            "sma200_stance": _stance(price, sma200),
            "rsi": _finite(last.get("rsi_14")),
            "macd_hist": _finite(last.get("hist")),
            "macd_fresh_cross": bool(last.get("fresh_cross", False)),
            "pct_b": _finite(last.get("pct_b")),
        }
    except indicators.InsufficientHistoryError:
        _LOGGER.info("price history too short for full indicators (%d bars)", len(close))

    week52 = frame.tail(config.WEEK52_BARS)
    week52_high, week52_low = _finite(week52["High"].max()), _finite(week52["Low"].min())
    ytd_pct = None
    last_date = frame.index[-1]
    prior_mask = frame.index.year == (last_date.year - 1)
    if prior_mask.any() and price is not None:
        prior_close = _finite(frame.loc[prior_mask, "Close"].iloc[-1])
        if prior_close not in (None, 0.0):
            ytd_pct = round((price / prior_close - 1.0) * 100.0, 2)

    rows = [
        {"date": idx.date().isoformat(),
         "open": round(_finite(row["Open"]) or 0.0, 2), "high": round(_finite(row["High"]) or 0.0, 2),
         "low": round(_finite(row["Low"]) or 0.0, 2), "close": round(_finite(row["Close"]) or 0.0, 2),
         "volume": int(row["Volume"])}
        for idx, row in frame.tail(5).iterrows()
    ]
    payload = {
        "ticker": args.ticker.upper(), "period": args.period,
        "current_price": price, "previous_close": prev_close,
        "week52_high": week52_high, "week52_low": week52_low, "ytd_pct": ytd_pct,
        "rows": rows, "indicator_state": indicator_state,
    }
    return make_result(True, data=payload, source="yfinance")

def price_digest(data: Dict[str, Any]) -> str:
    """Compact observation digest for the agent's context window."""
    state = data.get("indicator_state", {})
    return (f"price={data.get('current_price')} prev={data.get('previous_close')} "
            f"vs SMA50={state.get('sma50_stance')} vs SMA200={state.get('sma200_stance')} "
            f"RSI={state.get('rsi')} macd_hist={state.get('macd_hist')} "
            f"%B={state.get('pct_b')} 52w={data.get('week52_low')}-{data.get('week52_high')} "
            f"YTD={data.get('ytd_pct')}%")


# ── Tool 2: get_news ──────────────────────────────────────────────────────────
def _normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(title or "").lower())


def _normalize_pubdate(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError):
        return raw


def _normalize_yf_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Handle BOTH yfinance news shapes (legacy flat and nested under content)."""
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title") or ""
        provider = content.get("provider") or {}
        source = provider.get("displayName") if isinstance(provider, dict) else ""
        url = ""
        for holder in (content.get("canonicalUrl"), content.get("clickThroughUrl")):
            if isinstance(holder, dict) and holder.get("url"):
                url = str(holder["url"])
                break
        published = content.get("pubDate") or ""
    else:
        title = item.get("title") or ""
        source = item.get("publisher") or ""
        link = item.get("link") or ""
        if isinstance(link, dict):
            link = link.get("url") or ""
        url = str(link)
        ts = item.get("providerPublishTime")
        published = (datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
                     if isinstance(ts, (int, float)) and ts > 0
                     else str(item.get("publicationDate") or ""))
    title = html.unescape(str(title)).strip()
    if not title:
        return None
    return {"title": title, "source": str(source or "Yahoo Finance"),
            "published": _normalize_pubdate(str(published)), "url": url}

def get_news(ticker: str, n: int) -> ToolResult:
    """Ladder yfinance -> Yahoo RSS -> Google RSS; dedupe; coverage label."""
    try:
        args = GetNewsArgs(ticker=ticker, n=n)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"n must be between {config.NEWS_MIN_COUNT} and {config.NEWS_MAX_COUNT}",
                           source="news-ladder")
    sources: List[Dict[str, Any]] = [
        ("yfinance", lambda: [_normalize_yf_item(i) for i in _news_raw(args.ticker)
                              if _normalize_yf_item(i)]),
        ("yahoo_rss", lambda: _parse_rss(_http_get(config.YAHOO_RSS_URL.format(ticker=args.ticker)),
                                         "Yahoo Finance RSS")),
        ("google_rss", lambda: _parse_rss(_http_get(config.GOOGLE_RSS_URL.format(ticker=args.ticker)),
                                          "Google News")),
    ]
    seen: set = set()
    items: List[Dict[str, str]] = []
    used: List[str] = []
    for name, fetch in sources:
        if len(items) >= args.n:
            break
        try:
            batch = fetch()
        except Exception as exc:  # noqa: BLE001 - a bad source is skipped, not fatal
            _LOGGER.warning("news source %s failed: %s", name, exc)
            continue
        if batch:
            used.append(name)
        for entry in batch:
            key = _normalize_title_key(entry["title"])
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(entry)
            if len(items) >= args.n:
                break
    if not items:
        return make_result(False, data=[], error="no headlines from any source",
                           hint="use web_search for newswires", source="news-ladder")
    partial_min = max(2, round(args.n * config.NEWS_PARTIAL_SHARE))
    coverage = "full" if len(items) >= args.n else (
        "partial" if len(items) >= partial_min else "low")
    return make_result(True, data={"headlines": items, "coverage": coverage,
                                   "sources": used}, source="news-ladder")


def _parse_rss(xml_text: str, default_source: str) -> List[Dict[str, str]]:
    items = []
    for node in ET.fromstring(xml_text).iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": html.unescape(title),
            "source": (node.findtext("source") or default_source).strip() or default_source,
            "published": _normalize_pubdate(node.findtext("pubDate") or ""),
            "url": (node.findtext("link") or "").strip(),
        })
    return items


def news_digest(data: Dict[str, Any]) -> str:
    headlines = data.get("headlines", [])
    top = " | ".join(h["title"][:60] for h in headlines[:3])
    return f"{len(headlines)} headlines ({data.get('coverage')}): {top}"

# ── Tool 3: calculate_volatility ──────────────────────────────────────────────
def calculate_volatility(ticker: str, window: int) -> ToolResult:
    """Annualized realized volatility over the last ``window`` daily log returns."""
    try:
        args = CalculateVolatilityArgs(ticker=ticker, window=window)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"window must be between {config.VOL_WINDOW_MIN} "
                                f"and {config.VOL_WINDOW_MAX}",
                           source="yfinance")
    try:
        frame = _fetch_history(args.ticker, "2y")  # own fetch: never borrow another tool's state
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"{type(exc).__name__}: {exc}",
                           hint="try web_search for volatility commentary", source="yfinance")
    if frame.empty or "Close" not in frame:
        return make_result(False, error="empty price frame",
                           hint="try web_search for volatility commentary", source="yfinance")
    close = frame["Close"].astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < args.window:
        args.window = len(log_returns)  # degrade gracefully, report the real n_obs
    used = log_returns.tail(args.window)
    daily = float(used.std(ddof=0))
    annualized = daily * math.sqrt(config.TRADING_DAYS_PER_YEAR)
    if annualized < config.VOL_BAND_LOW_MAX:
        band = "low"
    elif annualized >= config.VOL_BAND_HIGH_MIN:
        band = "high"
    else:
        band = "moderate"
    payload = {
        "ticker": args.ticker.upper(), "annualized": round(annualized, 4),
        "daily": round(daily, 6), "window": args.window,
        "n_obs": int(len(used)), "band": band,
    }
    return make_result(True, data=payload, source="yfinance")


def volatility_digest(data: Dict[str, Any]) -> str:
    return (f"vol annualized={data.get('annualized')} ({data.get('band')}) "
            f"daily={data.get('daily')} window={data.get('window')} n={data.get('n_obs')}")

# ── Tool 4: llm_sentiment (ONE LLM call per headline) ─────────────────────────
def llm_sentiment(headlines: List[str], llm: LLMClient, ticker: str = "") -> ToolResult:
    """Score each headline separately, then aggregate with confidence weighting."""
    try:
        args = LlmSentimentArgs(headlines=headlines[:config.SENTIMENT_MAX_HEADLINES])
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint="headlines must be a non-empty list of strings",
                           source="llm")
    items: List[Dict[str, Any]] = []
    for headline in args.headlines:
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM},
            {"role": "user", "content": SENTIMENT_USER.format(
                ticker=ticker.upper(), headline=headline)},
        ]
        try:
            value = llm.complete_json(messages, HeadlineSentiment)
            value = value.model_copy(update={"headline": headline})
            items.append(value.model_dump())
        except Exception:  # noqa: BLE001 - one bad headline never kills the batch
            items.append({"headline": headline, "sentiment": "neutral", "confidence": 0.0,
                          "brief_reason": "sentiment unavailable; neutral sentinel",
                          "estimate": True})
    sign_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    weighted = sum(sign_map[item["sentiment"]] * item["confidence"] for item in items)
    total_confidence = sum(item["confidence"] for item in items)
    overall = weighted / total_confidence if total_confidence > 0 else 0.0
    overall = max(-1.0, min(1.0, overall))
    if overall > config.SENTIMENT_POSITIVE_THRESHOLD:
        label = "positive"
    elif overall < config.SENTIMENT_NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    counts = {name: sum(1 for item in items if item["sentiment"] == name)
              for name in ("positive", "negative", "neutral")}
    payload = {
        "items": items, "overall_score": round(overall, 4), "label": label,
        "counts": counts,
        "estimates": sum(1 for item in items if item.get("estimate")),
    }
    return make_result(True, data=payload, source="llm")


def sentiment_digest(data: Dict[str, Any]) -> str:
    return (f"sentiment={data.get('label')} ({data.get('overall_score'):+.3f}) "
            f"counts={data.get('counts')} estimates={data.get('estimates')}")


# ── Tool 5: web_search ────────────────────────────────────────────────────────
def web_search(query: str) -> ToolResult:
    """DuckDuckGo search with a politeness sleep and one backoff retry."""
    try:
        args = WebSearchArgs(query=query)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint="query must be at least 3 characters", source="duckduckgo")
    attempts = 0
    last_error: Optional[str] = None
    while attempts < 2:
        attempts += 1
        try:
            time.sleep(config.WEB_SEARCH_SLEEP_S * attempts)  # politeness, longer on retry
            raw = _ddgs_text(args.query, max_results=config.WEB_SEARCH_MAX_RESULTS)
            results = [
                {"title": str(item.get("title", ""))[:200],
                 "snippet": str(item.get("body", ""))[:300],
                 "url": str(item.get("href", ""))}
                for item in raw if isinstance(item, dict) and item.get("title")
            ]
            if not results:
                return make_result(False, data=[], error="no search results",
                                   hint="try a broader query", source="duckduckgo")
            return make_result(True, data={"query": args.query, "results": results},
                               source="duckduckgo")
        except Exception as exc:  # noqa: BLE001 - one backoff retry, then give up gracefully
            last_error = f"{type(exc).__name__}: {exc}"
            _LOGGER.warning("web_search attempt %d failed: %s", attempts, last_error)
    return make_result(False, error=last_error or "search failed",
                       hint="skip commentary and rely on other evidence",
                       source="duckduckgo")


def search_digest(data: Dict[str, Any]) -> str:
    results = data.get("results", [])
    return f"{len(results)} results: " + " | ".join(
        r["title"][:50] for r in results[:2])

# ── Registry and dispatch ─────────────────────────────────────────────────────
_DIGESTERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    config.TOOL_GET_PRICE_DATA: price_digest,
    config.TOOL_GET_NEWS: news_digest,
    config.TOOL_CALCULATE_VOLATILITY: volatility_digest,
    config.TOOL_LLM_SENTIMENT: sentiment_digest,
    config.TOOL_WEB_SEARCH: search_digest,
}

_TOOL_RUNNERS: Dict[str, Callable[..., ToolResult]] = {
    config.TOOL_GET_PRICE_DATA: get_price_data,
    config.TOOL_GET_NEWS: get_news,
    config.TOOL_CALCULATE_VOLATILITY: calculate_volatility,
    config.TOOL_LLM_SENTIMENT: llm_sentiment,
    config.TOOL_WEB_SEARCH: web_search,
}


def tool_signature(name: str) -> str:
    """One-line signature used in the agent prompts."""
    signatures = {
        config.TOOL_GET_PRICE_DATA: "get_price_data(ticker: str, period: str)",
        config.TOOL_GET_NEWS: "get_news(ticker: str, n: int)",
        config.TOOL_CALCULATE_VOLATILITY: "calculate_volatility(ticker: str, window: int)",
        config.TOOL_LLM_SENTIMENT: "llm_sentiment(headlines: list[str])",
        config.TOOL_WEB_SEARCH: "web_search(query: str)",
    }
    return signatures.get(name, name)


def allowed_tools_block(whitelist) -> str:
    """Render the allowed tools (with signatures) for an agent prompt."""
    return "\n".join(f"- {tool_signature(name)}" for name in whitelist)


def dispatch(name: str, args: Dict[str, Any], llm: Optional[LLMClient],
             ticker: str = "") -> ToolResult:
    """Validate args, run the tool, and guarantee a ToolResult comes back."""
    if name not in _TOOL_RUNNERS:
        return make_result(False, error=f"unknown tool {name!r}",
                           hint=f"known tools: {', '.join(sorted(_TOOL_RUNNERS))}",
                           source="dispatcher")
    model = ARG_MODELS[name]
    try:
        validated = model.model_validate(args)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=str(exc.errors()[0].get("input", ""))[:120],
                           source="dispatcher")
    runner = _TOOL_RUNNERS[name]
    if name == config.TOOL_LLM_SENTIMENT:
        if llm is None:
            return make_result(False, error="LLM client unavailable",
                               hint="configure LLM_* environment variables", source="llm")
        return runner(validated.headlines, llm, ticker=ticker)
    if name == config.TOOL_GET_PRICE_DATA:
        return runner(validated.ticker, validated.period)
    if name == config.TOOL_GET_NEWS:
        return runner(validated.ticker, validated.n)
    if name == config.TOOL_CALCULATE_VOLATILITY:
        return runner(validated.ticker, validated.window)
    if name == config.TOOL_WEB_SEARCH:
        return runner(validated.query)
    return make_result(False, error="unroutable tool", source="dispatcher")


def digest_for(name: str, result: ToolResult) -> str:
    """One-line observation digest; prefixed with the failure when not ok."""
    if not result.ok:
        return f"FAILED: {result.error or 'unknown error'} (hint: {result.hint})"
    digester = _DIGESTERS.get(name)
    if digester is None:
        return json.dumps(result.data, ensure_ascii=False, default=str)
    try:
        return digester(result.data or {})
    except Exception:  # noqa: BLE001 - a digest must never break the loop
        return json.dumps(result.data, ensure_ascii=False, default=str)
