# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'session memory store plus persistent per-ticker-per-day cache with corruption guard', Date: 2026-10-06
"""Memory: short-term session store and a persistent per-ticker/per-day cache.

The session store holds every ToolResult, the briefs, the clarification and the
final report; answering a follow-up from the store must never trigger tool
calls, so the store also owns the tool-call counter. The persistent cache is
keyed ``{TICKER}_{YYYY-MM-DD}.json`` (date from the clock, never a literal) and
is guarded against corruption: a parse failure invalidates the file and the
next run proceeds fully.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from . import config

_LOGGER = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1


class MemoryStore:
    """Session-scoped memory for one research run."""

    def __init__(self) -> None:
        self.tool_results: Dict[str, Dict[str, Any]] = {}  # "tool|args-json" -> ToolResult dict
        self.briefs: Dict[str, Dict[str, Any]] = {}        # "brief_v1" / "brief_v2"
        self.clarification: Optional[Dict[str, Any]] = None
        self.report: Optional[Dict[str, Any]] = None
        self.tool_call_count = 0
        self.replans = 0
        self.critique_cycles = 0
        self.memory_hits = 0

    def add_tool_result(self, agent: str, tool: str, args: Dict[str, Any],
                        result: Dict[str, Any]) -> str:
        key = f"{agent}|{tool}|{json.dumps(args, sort_keys=True)}"
        self.tool_results[key] = {"agent": agent, "tool": tool, "args": args, "result": result}
        self.tool_call_count += 1
        return key

    def snapshot(self) -> Dict[str, Any]:
        """Serializable view handed to the memory answerer."""
        return {
            "tool_results": list(self.tool_results.values()),
            "briefs": self.briefs,
            "clarification": self.clarification,
            "report": self.report,
        }


# The most recent session (used by the public answer_from_memory entry point).
LAST_SESSION: Optional[MemoryStore] = None


def cache_path_for(ticker: str, cache_dir: Path = config.CACHE_DIR,
                   day: Optional[date] = None) -> Path:
    """``cache/{TICKER}_{YYYY-MM-DD}.json`` — the date comes from the clock."""
    day = day or date.today()
    return Path(cache_dir) / f"{ticker.upper()}_{day.isoformat()}.json"


def save_cache(payload: Dict[str, Any], ticker: str,
               cache_dir: Path = config.CACHE_DIR) -> Path:
    """Persist the run artefacts keyed by ticker and day."""
    path = cache_path_for(ticker, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": CACHE_SCHEMA_VERSION, **payload}
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    _LOGGER.info("cache saved: %s", path.name)
    return path


def load_cache(ticker: str, cache_dir: Path = config.CACHE_DIR) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load today's cache for the ticker.

    Returns ``(payload, None)`` on a hit, ``(None, None)`` on a miss, and
    ``(None, error)`` when the file exists but is corrupt (the caller
    invalidates it and runs fully).
    """
    path = cache_path_for(ticker, cache_dir)
    if not path.exists():
        return None, None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOGGER.warning("corrupt cache file %s: %s - invalidating", path.name, exc)
        try:
            path.unlink()
        except OSError:
            pass
        return None, f"corrupt cache file: {exc}"
    if not isinstance(document, dict) or document.get("schema_version") != CACHE_SCHEMA_VERSION:
        _LOGGER.warning("cache %s has an unknown schema - invalidating", path.name)
        return None, "unknown schema version"
    return document, None
