# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'news tool with ladder, both yfinance shapes, rss parsing, dedupe and coverage label', Date: 2026-10-06
"""get_news(ticker, n): headlines with a source ladder and dedupe."""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..schemas import ToolResult
from . import sources
from .base import make_result

_LOGGER = logging.getLogger(__name__)


class GetNewsArgs(BaseModel):
    ticker: str = Field(min_length=1)
    n: int = Field(default=config.NEWS_DEFAULT_COUNT, ge=config.NEWS_MIN_COUNT,
                   le=config.NEWS_MAX_COUNT)


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


def get_news(ticker: str, n: int) -> ToolResult:
    """Ladder yfinance -> Yahoo RSS -> Google RSS; dedupe; coverage label."""
    try:
        args = GetNewsArgs(ticker=ticker, n=n)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"n must be between {config.NEWS_MIN_COUNT} and {config.NEWS_MAX_COUNT}",
                           source="news-ladder")
    ladder: List[tuple] = [
        ("yfinance", lambda: [_normalize_yf_item(i) for i in sources._news_raw(args.ticker)
                              if _normalize_yf_item(i)]),
        ("yahoo_rss", lambda: _parse_rss(sources._http_get(config.YAHOO_RSS_URL.format(ticker=args.ticker)),
                                         "Yahoo Finance RSS")),
        ("google_rss", lambda: _parse_rss(sources._http_get(config.GOOGLE_RSS_URL.format(ticker=args.ticker)),
                                          "Google News")),
    ]
    seen = set()
    items: List[Dict[str, str]] = []
    used: List[str] = []
    for name, fetch in ladder:
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


def news_digest(data: Dict[str, Any]) -> str:
    headlines = data.get("headlines", [])
    top = " | ".join(h["title"][:60] for h in headlines[:3])
    return f"{len(headlines)} headlines ({data.get('coverage')}): {top}"
