"""Presentation layer: renders the JSONL trace as a readable narrative.

This is the ONLY module that formats output for humans; everything inside
``src`` uses logging instead.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TRACE = Path(__file__).resolve().parents[1] / "logs" / "agent_trace.jsonl"

def load_trace(path: Optional[Path] = None,
               run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read the trace file (optionally filtered to one run)."""
    trace_path = Path(path) if path else DEFAULT_TRACE
    entries = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if run_id is None or entry.get("run_id") == run_id:
            entries.append(entry)
    return entries

def _as_dict(value: Any) -> Dict[str, Any]:
    """Trace args fields are JSON strings; coerce them for rendering."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}

def render_narrative(entries: List[Dict[str, Any]]) -> str:
    """Readable step-by-step narrative of one run."""
    lines: List[str] = []
    for entry in entries:
        kind = entry.get("kind")
        agent = entry.get("agent", "?")
        if kind == "tool":
            status = "ok" if entry.get("ok") else f"FAILED ({entry.get('error')})"
            lines.append(f"[{agent}] {entry.get('tool')}({entry.get('args')}) -> {status}")
            if entry.get("output"):
                lines.append(f"    observe: {str(entry['output'])[:160]}")
        elif kind == "replan":
            args = _as_dict(entry.get("args"))
            lines.append(f"[{agent}] REPLAN: {entry.get('output')} "
                         f"({args.get('from')} -> {args.get('to')})")
        elif kind == "critique":
            lines.append(f"[{agent}] CRITIQUE REQUEST: {str(entry.get('output'))[:200]}")
        elif kind == "handoff":
            lines.append(f"[{agent}] HANDOFF {entry.get('tool')}: "
                         f"{str(entry.get('output'))[:160]}")
        elif kind == "memory":
            lines.append(f"[{agent}] MEMORY: {str(entry.get('output'))[:160]}")
        elif kind == "cache":
            lines.append(f"[cache] {str(entry.get('output'))[:160]}")
        elif kind == "llm":
            hit = " (cache)" if entry.get("cache_hit") else ""
            lines.append(f"[{agent}] decide{hit}: {entry.get('tool')}")
    return "\n".join(lines)

def summarize(entries: List[Dict[str, Any]]) -> str:
    """Summary counts: tool calls per agent, replans, critiques, memory hits."""
    tools_per_agent = dict(Counter(
        entry.get("agent") for entry in entries if entry.get("kind") == "tool"
    ))
    replans = sum(1 for entry in entries if entry.get("kind") == "replan")
    critiques = sum(1 for entry in entries if entry.get("kind") == "critique")
    memory = sum(1 for entry in entries if entry.get("kind") == "memory")
    cache = sum(1 for entry in entries if entry.get("kind") == "cache")
    failures = sum(1 for entry in entries
                   if entry.get("kind") == "tool" and not entry.get("ok"))
    return "\n".join([
        "trace summary:",
        f"  tool calls per agent: {tools_per_agent}",
        f"  replans: {replans} | critique events: {critiques} | "
        f"memory events: {memory} | cache events: {cache}",
        f"  failed tool calls: {failures}",
    ])

def render_file(path: Optional[Path] = None,
                run_id: Optional[str] = None) -> str:
    """Narrative + summary for a trace file."""
    entries = load_trace(path, run_id)
    return render_narrative(entries) + "\n\n" + summarize(entries)
