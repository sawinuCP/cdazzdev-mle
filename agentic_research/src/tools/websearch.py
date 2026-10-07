# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'web search tool with politeness sleep and one backoff retry', Date: 2026-10-06
"""web_search(query): DuckDuckGo commentary snippets."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..schemas import ToolResult
from . import sources
from .base import make_result

_LOGGER = logging.getLogger(__name__)


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=3)


def web_search(query: str) -> ToolResult:
    """DuckDuckGo search with a politeness sleep and one backoff retry."""
    try:
        args = WebSearchArgs(query=query)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint="query must be at least 3 characters", source="duckduckgo")
    attempts = 0
    last_error = None
    while attempts < 2:
        attempts += 1
        try:
            time.sleep(config.WEB_SEARCH_SLEEP_S * attempts)  # politeness, longer on retry
            raw = sources._ddgs_text(args.query, max_results=config.WEB_SEARCH_MAX_RESULTS)
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
