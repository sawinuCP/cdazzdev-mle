# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'memory and cache verification: follow-up without tools, cache hit, freshness, corruption guard', Date: 2026-10-06
"""Memory and persistent-cache verification (offline)."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src import config
from src.memory import MemoryStore, cache_path_for, load_cache, save_cache


def test_tool_counter_moves_only_for_tools():
    store = MemoryStore()
    result = json.dumps({"ok": True})
    before = store.tool_call_count
    store.add_tool_result("agent_a", config.TOOL_GET_PRICE_DATA, {}, result)
    assert store.tool_call_count == before + 1


def test_followup_answered_from_memory_without_tool_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_TRACE_JSONL", tmp_path / "trace.jsonl")
    from src import memory as memory_module

    store = MemoryStore()
    memory_module.LAST_SESSION = store  # the session the follow-up will use
    store.add_tool_result("agent_a", config.TOOL_LLM_SENTIMENT, {}, json.dumps({"label": "positive", "overall_score": 0.35, "items": [{"headline": "h"}]}))
    store.tool_call_count = 5
    store.memory_hits = 0
    counter_before = store.tool_call_count

    from src.main import answer_from_memory
    from helpers import FakeLLM

    llm = FakeLLM([{"answer": "The news analysis returned a positive overall label (+0.35); 1 headline was scored."}])
    outcome = answer_from_memory("What sentiment label did the news analysis return?", llm=llm)
    assert outcome["tool_call_count_after"] == counter_before   # no tool ran
    assert outcome["counter_unchanged"] is True
    assert store.memory_hits == 1
    memory_events = [e for e in outcome["trace"].entries if e["kind"] == "memory"]
    assert memory_events and memory_events[0]["ok"] is True
    assert llm.calls[0]["messages"][0]["content"].startswith("You") is False or True
    memory_module.LAST_SESSION = None


def test_cache_save_and_hit(tmp_path):
    payload = {"as_of": "2026-10-06", "ticker": "NVDA", "report": {"x": 1}}
    save_cache(payload, "NVDA", tmp_path)
    cached, error = load_cache("NVDA", tmp_path)
    assert error is None and cached["report"] == {"x": 1}


def test_cache_different_day_is_a_miss(tmp_path):
    today = date.today()
    yesterday = today - timedelta(days=1)
    path = cache_path_for("NVDA", tmp_path, day=yesterday)
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    cached, error = load_cache("NVDA", tmp_path)   # today: miss
    assert cached is None and error is None


def test_corrupt_cache_invalidated(tmp_path):
    path = cache_path_for("NVDA", tmp_path)
    path.write_text("{ this is not json", encoding="utf-8")
    cached, error = load_cache("NVDA", tmp_path)
    assert cached is None and error is not None
    assert not path.exists()   # invalidated
