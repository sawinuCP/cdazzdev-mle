# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'graph end-to-end verification with mocked tools and scripted LLM plus crash-injection', Date: 2026-10-06
"""Graph end-to-end verification (mocked data sources, scripted LLM)."""
from __future__ import annotations

import json

import pytest

from src import config
from src.agents.graph import run_research
from src.schemas import ResearchReport
from helpers import FakeLLM, synthetic_frame


@pytest.fixture(autouse=True)
def mock_sources(monkeypatch, tmp_path):
    frame = synthetic_frame()
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "LOG_TRACE_JSONL", tmp_path / "agent_trace.jsonl")
    monkeypatch.setattr(config, "LLM_CACHE_JSON", tmp_path / ".llm_cache.json")
    monkeypatch.setattr("src.runtime.tracing.config", config)
    monkeypatch.setattr("src.runtime.memory.config", config)
    monkeypatch.setattr(config, "FAULT_INJECT", {})
    monkeypatch.setattr("src.tools.sources._fetch_history", lambda t, p: frame)
    monkeypatch.setattr("src.tools.sources._news_raw", lambda t: [])
    monkeypatch.setattr(
        "src.tools.sources._http_get",
        lambda url: ('<?xml version="1.0"?><rss version="2.0"><channel>'
                     "<item><title>Mock headline</title><link>http://x</link></item>"
                     "</channel></rss>"))
    monkeypatch.setattr("src.tools.sources._ddgs_text",
                        lambda q, max_results: [
                            {"title": "Commentary", "body": "b", "href": "http://x"}])
    return frame


def _scripted_llm() -> FakeLLM:
    """Payload order: A loop, B loop, critique writer, sentiment, composer."""
    return FakeLLM([
        {"thought": "need price", "action": config.TOOL_GET_PRICE_DATA,
         "args": {"ticker": "NVDA", "period": "1y"}, "replan_reason": None},
        {"thought": "need vol", "action": config.TOOL_CALCULATE_VOLATILITY,
         "args": {"ticker": "NVDA", "window": 90}, "replan_reason": None},
        {"thought": "quant complete", "action": "finish"},
        {"thought": "need news", "action": config.TOOL_GET_NEWS,
         "args": {"ticker": "NVDA", "n": 5}, "replan_reason": None},
        {"thought": "need commentary", "action": config.TOOL_WEB_SEARCH,
         "args": {"query": "NVDA outlook"}, "replan_reason": None},
        {"thought": "gather complete", "action": "finish"},
        {"request_id": "crit-1",
         "questions": [
             {"question_id": "q1", "text": "Score the attached headlines for market sentiment."},
             {"question_id": "q2", "text": "Provide the 30-day annualized volatility for comparison."},
         ],
         "payload": {"headlines": ["Mock headline"]}},
        {"headline": "Mock headline", "sentiment": "positive", "confidence": 0.8,
         "brief_reason": "r", "estimate": False},
        {"ticker": "NVDA",
         "financial_health_summary": {"narrative": "Positive trend with high valuation.",
                                      "evidence_refs": ["get_price_data", "calculate_volatility"]},
         "risks": [
             {"title": "Volatility", "detail": "moderate band", "likelihood": 3,
              "evidence": [{"source_tool": "calculate_volatility", "key_datum": "ann 0.4",
                            "url": None}]},
             {"title": "Valuation", "detail": "extended", "likelihood": 3,
              "evidence": [{"source_tool": "get_price_data", "key_datum": "%B 0.9",
                            "url": None}]},
             {"title": "Sentiment shift", "detail": "news-driven", "likelihood": 2,
              "evidence": [{"source_tool": "llm_sentiment", "key_datum": "positive 0.35",
                            "url": None}]},
         ],
         "hedge": {"strategy": "Collar", "instruments": ["protective put"],
                   "rationale": "1-sigma band sized",
                   "data_basis": ["1-sigma 90-day band: ±10.2", "price 121.5"],
                   "cost_note": "n/a"},
         "meta_note": "done"},
    ])

def test_full_two_agent_run(tmp_path):
    llm = _scripted_llm()
    result = run_research("NVDA", llm=llm, use_cache=False)
    assert result["error"] == ""                      # zero unhandled exceptions
    report = ResearchReport.model_validate(result["report"])
    assert report.meta.run_id == result["run_id"]
    assert len(report.risks) == 3

    trace = result["trace"]
    kinds = [entry["kind"] for entry in trace.entries]
    assert kinds.count("critique") == 1               # exactly one critique cycle
    assert "handoff" in kinds and "cache" in kinds    # cache events recorded
    # critique loop visible: brief v2 has the sentiment filled
    saved = (config.CACHE_DIR / f"NVDA_{__import__('datetime').date.today().isoformat()}.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["brief_v2"]["sentiment"] is not None
    assert payload["meta"]["critique_cycles"] == 1
    # every trace line carries the required fields
    for entry in trace.entries:
        for field in ("ts_utc", "run_id", "kind", "agent", "tool", "args",
                      "output", "duration_ms", "ok", "error"):
            assert field in entry


def test_fault_injection_stamps_replan(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAULT_INJECT", {config.TOOL_GET_NEWS: "empty"})
    llm = _scripted_llm()
    result = run_research("NVDA", llm=llm, use_cache=False)
    assert result["error"] == ""
    replans = [e for e in result["trace"].entries if e["kind"] == "replan"]
    assert replans, "expected at least one visible replan under fault injection"
    assert any("failed" in e["output"] or "changed" in e["output"] for e in replans)


def test_crash_injection_returns_degraded(tmp_path, monkeypatch):
    import src.agents.graph as graph_module

    def exploding(ticker, llm, trace, store):
        raise RuntimeError("injected crash")

    monkeypatch.setattr(graph_module, "run_agent_a", exploding)  # the name graph uses
    llm = _scripted_llm()
    result = run_research("NVDA", llm=llm, use_cache=False)
    assert result["degraded"] is True                 # the wrapper caught it
    assert result["report"] is not None               # minimal valid artefact returned
    assert "agent_a_brief" in result["error"] and "RuntimeError: injected crash" in result["error"]
