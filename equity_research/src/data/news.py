# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'news retrieval ladder yfinance both shapes plus RSS fallbacks with dedupe', Date: 2026-10-06
"""Headline retrieval ladder: yfinance -> Yahoo Finance RSS -> Google News RSS.

Every network path is individually wrapped: a failing source is logged and
skipped, and the caller always receives a best-effort result with a coverage
label instead of an exception.
"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Tuple

import requests

from .. import config

_LOGGER = logging.getLogger(__name__)


def _normalize_title_key(title: str) -> str:
    """Normalization key for dedupe: unescaped, lowercased, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", html.unescape(title or "").lower())


def _normalize_pubdate(raw: str) -> str:
    """Best-effort publication date -> ISO date string ("" when unknown)."""
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


def _normalize_yf_item(item: Dict[str, Any]) -> Dict[str, str] | None:
    """Normalize either yfinance news shape.

    yfinance changed its payload over time: older releases return flat dicts
    (``title``, ``publisher``, ``link``, ``providerPublishTime``); newer ones
    nest everything under ``content`` (``content.title``,
    ``content.provider.displayName``, ``content.canonicalUrl.url``,
    ``content.pubDate``). Coding for both avoids a silent zero.
    """
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if isinstance(content, dict):  # newer nested shape
        title = content.get("title") or ""
        provider = content.get("provider") or {}
        source = provider.get("displayName") if isinstance(provider, dict) else ""
        url = ""
        for holder in (content.get("canonicalUrl"), content.get("clickThroughUrl")):
            if isinstance(holder, dict) and holder.get("url"):
                url = str(holder["url"])
                break
        published = content.get("pubDate") or ""
    else:  # legacy flat shape
        title = item.get("title") or ""
        source = item.get("publisher") or ""
        link = item.get("link") or ""
        if isinstance(link, dict):  # some builds return {"url": ...}
            link = link.get("url") or ""
        url = str(link)
        ts = item.get("providerPublishTime")
        if isinstance(ts, (int, float)) and ts > 0:
            published = datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
        else:
            published = str(item.get("publicationDate") or "")
    title = html.unescape(str(title)).strip()
    if not title:
        return None
    return {
        "title": title,
        "source": str(source or "Yahoo Finance"),
        "published": _normalize_pubdate(str(published)),
        "url": url,
    }

def _fetch_yfinance(ticker: str) -> List[Dict[str, str]]:
    """Source 1: the yfinance news endpoint (both payload shapes)."""
    import yfinance as yf  # imported lazily so unit tests can stub the module

    raw = yf.Ticker(ticker).news or []
    normalized: List[Dict[str, str]] = []
    for item in raw:
        entry = _normalize_yf_item(item)
        if entry:
            normalized.append(entry)
    return normalized


def _parse_rss(xml_text: str, default_source: str) -> List[Dict[str, str]]:
    """Parse an RSS 2.0 document into normalized headline dicts."""
    items: List[Dict[str, str]] = []
    root = ET.fromstring(xml_text)
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": html.unescape(title),
                "source": (node.findtext("source") or default_source).strip() or default_source,
                "published": _normalize_pubdate(node.findtext("pubDate") or ""),
                "url": (node.findtext("link") or "").strip(),
            }
        )
    return items


def _http_get(url: str) -> str:
    """Single guarded HTTP GET with the configured timeout and user agent."""
    response = requests.get(
        url,
        timeout=config.HTTP_TIMEOUT_S,
        headers={"User-Agent": config.USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def _fetch_yahoo_rss(ticker: str) -> List[Dict[str, str]]:
    """Source 2: Yahoo Finance per-ticker headline RSS feed."""
    return _parse_rss(_http_get(config.YAHOO_RSS_URL.format(ticker=ticker)), "Yahoo Finance RSS")


def _fetch_google_rss(ticker: str) -> List[Dict[str, str]]:
    """Source 3: Google News search RSS (last resort)."""
    return _parse_rss(_http_get(config.GOOGLE_RSS_URL.format(ticker=ticker)), "Google News")


def fetch_headlines(
    ticker: str,
    min_count: int = config.MIN_HEADLINES,
    cap: int = config.MAX_HEADLINES,
) -> Dict[str, Any]:
    """Run the ladder until ``min_count`` unique headlines (capped at ``cap``).

    Returns ``{"headlines": [...], "coverage": full|partial|low, "sources": [...]}``;
    never raises for source failures (they are logged and skipped).
    """
    ladder: List[Tuple[str, Callable[[str], List[Dict[str, str]]]]] = [
        ("yfinance", _fetch_yfinance),
        ("yahoo_rss", _fetch_yahoo_rss),
        ("google_rss", _fetch_google_rss),
    ]
    seen: set[str] = set()
    collected: List[Dict[str, str]] = []
    used_sources: List[str] = []
    for name, fetcher in ladder:
        if len(collected) >= min_count:
            break
        try:
            batch = fetcher(ticker)
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the run
            _LOGGER.warning("news source %s failed: %s", name, exc)
            continue
        if batch:  # record only sources that actually contributed headlines
            used_sources.append(name)
        for item in batch:
            key = _normalize_title_key(item["title"])
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append(item)
            if len(collected) >= cap:
                break
    if len(collected) >= min_count:
        coverage = "full"
    elif len(collected) >= config.PARTIAL_COVERAGE_MIN:
        coverage = "partial"
    else:
        coverage = "low"
    _LOGGER.info(
        "news: %d unique headlines for %s (coverage=%s, sources=%s)",
        len(collected), ticker, coverage, ",".join(used_sources) or "none",
    )
    return {"headlines": collected, "coverage": coverage, "sources": used_sources}
