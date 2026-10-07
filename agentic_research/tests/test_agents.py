"""Offline verification for the agent loop (mocked LLM + data sources)."""
from __future__ import annotations

import json

import pytest

from src import config, tools
from src.agents.agents import AgentLoop
from src.runtime.memory import MemoryStore
from src.runtime.tracing import TraceLogger
from helpers import FakeLLM, synthetic_frame

@pytest.fixture
def store():
    return MemoryStore()

@pytest.fixture
def trace(tmp_path):
    return TraceLogger(path=tmp_path / "trace.jsonl")

@pytest.fixture(autouse=True)
def mock_data_sources(monkeypatch):
    """All network sources mocked; llm_sentiment stubbed to a canned result."""
    frame = synthetic_frame()
    monkeypatch.setattr(tools.sources, "_fetch_history", lambda t, p: frame)
    monkeypatch.setattr(tools.sources, "_news_raw", lambda t: [])
    monkeypatch.setattr(
        tools.sources, "_http_get",
        lambda url: ('<?xml version="1.0"?><rss version="2.0"><channel>'
                     "<item><title>Headline one</title><link>http://x</link></item>"
                     "</channel></rss>"))
    monkeypatch.setattr(tools.sources, "_ddgs_text",
                        lambda q, max_results: [
                            {"title": "Commentary", "body": "b", "href": "http://x"}])
    monkeypatch.setattr(
        tools, "llm_sentiment",
        lambda headlines, llm, ticker="": tools.make_result(
            True, data={"items": [{"headline": h, "sentiment": "neutral",
                                   "confidence": 0.9, "brief_reason": "r",
                                   "estimate": False} for h in headlines],
                        "overall_score": 0.0, "label": "neutral",
                        "counts": {"positive": 0, "negative": 0,
                                   "neutral": len(headlines)},
                        "estimates": 0},
            source="llm"))

def _llm_scripted(actions: list) -> FakeLLM:
    return FakeLLM(actions)

def _action(tool: str, args: dict = None, thought: str = "t",
            replan_reason: str = None) -> dict:
    return {"thought": thought, "action": tool, "args": args or {},
            "replan_reason": replan_reason}

def test_agent_chooses_its_own_order_a(trace, store):
    llm = _llm_scripted([
        _action(config.TOOL_GET_PRICE_DATA, {"ticker": "NVDA", "period": "1y"}),
        _action(config.TOOL_CALCULATE_VOLATILITY, {"ticker": "NVDA", "window": 90}),
        _action("finish", thought="done"),
    ])
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    executed = [o["tool"] for o in summary["observations"] if "ok" in o and o.get("tool") != "finish"]
    assert executed[0] == config.TOOL_GET_PRICE_DATA
    assert executed[1] == config.TOOL_CALCULATE_VOLATILITY

def test_agent_chooses_a_different_order_b(trace, store):
    """ TWO different scripted orders both work - the order is not hardcoded."""
    llm = _llm_scripted([
        _action(config.TOOL_WEB_SEARCH, {"query": "NVDA analyst commentary"}),
        _action(config.TOOL_GET_NEWS, {"ticker": "NVDA", "n": 10}),
        _action("finish", thought="done"),
    ])
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    executed = [o["tool"] for o in summary["observations"]]
    assert executed == [config.TOOL_WEB_SEARCH, config.TOOL_GET_NEWS]

def test_duplicate_call_warns_then_is_refused(trace, store):
    llm = _llm_scripted([
        _action(config.TOOL_GET_NEWS, {"ticker": "NVDA", "n": 10}),
        _action(config.TOOL_GET_NEWS, {"ticker": "NVDA", "n": 10}),
        _action(config.TOOL_GET_NEWS, {"ticker": "NVDA", "n": 10}),
        _action("finish"),
    ])
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    digests = [o["digest"] for o in summary["observations"]]
    assert any("REFUSED" in d and "repeated identical call" in d for d in digests)

def test_iteration_cap_stops_the_loop(trace, store):
    llm = FakeLLM([_action(config.TOOL_GET_PRICE_DATA, {"period": "1y"}),
                   _action(config.TOOL_GET_NEWS, {"n": 10})])  # no finish
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, max_steps=2, objective="objective")
    summary = loop.run()
    assert summary["finished"] is False
    assert trace.entries[-1]["kind"] == "tool"  # stopped mid-work at the cap

def test_replan_stamped_from_llm_reason(trace, store):
    llm = _llm_scripted([
        _action(config.TOOL_GET_PRICE_DATA, {"ticker": "NVDA", "period": "1y"},
                thought="price fetched"),
        _action(config.TOOL_WEB_SEARCH, {"query": "NVDA risks"},
                thought="need risks", replan_reason="price snapshot changed my plan"),
        _action("finish"),
    ])
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    assert len(summary["replans"]) == 1
    assert "changed my plan" in summary["replans"][0]["reason"]
    assert summary["replans"][0]["to"] == config.TOOL_WEB_SEARCH

def test_replan_stamped_automatically_after_failed_observation(trace, store):
    llm = _llm_scripted([
        _action(config.TOOL_GET_PRICE_DATA, {"period": "1y"},
                thought="need price"),
        _action(config.TOOL_WEB_SEARCH, {"query": "NVDA price"},
                thought="fallback to commentary"),
        _action("finish"),
    ])
    monkey_frame = {"fetch_fails": True}
    import helpers  # noqa: F401 - ensure module is loaded for patching below
    # patched inside the test body because the fixture mocks success:
    import src.tools.sources as tools_module

    original_fetch = tools_module._fetch_history

    def failing_fetch(t, p):
        if monkey_frame["fetch_fails"]:
            raise RuntimeError("network down")
        return original_fetch(t, p)

    tools_module._fetch_history = failing_fetch  # direct patch; restored after test
    try:
        loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                         store, config.SINGLE_AGENT_MAX_STEPS, "objective")
        summary = loop.run()
    finally:
        tools_module._fetch_history = original_fetch
    reasons = [r["reason"] for r in summary["replans"]]
    assert any("previous action 'get_price_data' failed" in reason for reason in reasons)

def test_malformed_action_is_retried_then_skipped(trace, store):
    llm = _llm_scripted([
        AssertionError("unused"),
    ])
    llm.payloads = [
        AssertionError("never used"),
        _action(config.TOOL_GET_NEWS, {"n": 10}),
        _action("finish"),
    ]
    # first LLM reply is malformed (invalid JSON) via an exception:
    from src.runtime.llm_client import LLMValidationError

    llm.payloads[0] = LLMValidationError("invalid JSON")
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    assert any("not a valid action object" in w for w in loop.warnings) or \
        any(entry["tool"] == config.TOOL_GET_NEWS for entry in summary["observations"])
    assert summary["finished"] is True  # the loop recovered and reached finish

def test_fault_injection_makes_the_tool_fail(trace, store, monkeypatch):
    monkeypatch.setattr(config, "FAULT_INJECT", {config.TOOL_GET_NEWS: "empty"})
    llm = _llm_scripted([
        _action(config.TOOL_GET_NEWS, {"n": 10}),
        _action(config.TOOL_WEB_SEARCH, {"query": "NVDA newswires"}),
        _action("finish"),
    ])
    loop = AgentLoop("single", "NVDA", config.WHITELISTS["single"], llm, trace,
                     store, config.SINGLE_AGENT_MAX_STEPS, "objective")
    summary = loop.run()
    failed = [o for o in summary["observations"] if not o["ok"]]
    assert failed and failed[0]["tool"] == config.TOOL_GET_NEWS
    assert "injected fault" in failed[0]["digest"]
