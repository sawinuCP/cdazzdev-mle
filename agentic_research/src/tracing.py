# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'trace logger appending every tool/llm/memory/cache/replan event to JSONL', Date: 2026-10-06
"""Observability: one JSON line per event, written immediately.

``TraceLogger`` wraps every tool call and records LLM, memory, cache, replan
and critique events. Lines carry the tool name, arguments, the first 200
characters of the output and the wall-clock duration, so any claim about agent
behaviour can be verified from the committed file.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from . import config

TResult = TypeVar("TResult")

_EVENT_FIELDS = (
    "ts_utc", "run_id", "kind", "agent", "tool", "args",
    "output", "duration_ms", "ok", "error", "cache_hit",
)


class TraceLogger:
    """Thread-safe JSONL trace writer."""

    def __init__(self, path: Path = config.LOG_TRACE_JSONL, run_id: Optional[str] = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.entries: List[Dict[str, Any]] = []  # in-memory mirror for the notebook
        config.ensure_dirs()

    def log(self, kind: str, agent: str = "", tool: str = "", args: Any = None,
            output: Any = "", duration_ms: float = 0.0, ok: bool = True,
            error: Optional[str] = None, cache_hit: Optional[bool] = None) -> None:
        entry = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "kind": kind,
            "agent": agent,
            "tool": tool,
            "args": _truncate_json(args),
            "output": _truncate_text(output),
            "duration_ms": round(duration_ms, 2),
            "ok": ok,
            "error": _truncate_text(error or ""),
            "cache_hit": cache_hit,
        }
        with self.lock:
            self.entries.append(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def timed_tool_call(
        self, agent: str, tool: str, args: Any, fn: Callable[[], TResult]
    ) -> Tuple[TResult, Dict[str, Any]]:
        """Run one tool invocation, time it, and trace the outcome."""
        started = time.perf_counter()
        try:
            result = fn()
            duration_ms = (time.perf_counter() - started) * 1000.0
            self.log("tool", agent=agent, tool=tool, args=args,
                     output=_digest(result), duration_ms=duration_ms, ok=True)
            return result, {"ok": True, "duration_ms": duration_ms, "error": None}
        except Exception as exc:  # noqa: BLE001 - tools must not raise, but guard anyway
            duration_ms = (time.perf_counter() - started) * 1000.0
            self.log("tool", agent=agent, tool=tool, args=args, output="",
                     duration_ms=duration_ms, ok=False, error=f"{type(exc).__name__}: {exc}")
            raise


def _truncate_text(value: Any, limit: int = config.TRACE_OUTPUT_MAX_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit]


def _truncate_json(value: Any, limit: int = config.TRACE_OUTPUT_MAX_CHARS) -> str:
    return _truncate_text(value, limit)


def _digest(result: Any) -> str:
    """Compact digest of a ToolResult for the trace output field."""
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    else:
        payload = result
    return json.dumps(payload, ensure_ascii=False, default=str)
