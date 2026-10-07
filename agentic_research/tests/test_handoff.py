"""Handoff verification: typed brief, lossless round trip, tool-built numbers."""
from __future__ import annotations

import pytest

from src import config, tools
from src.agents.agents import build_brief_v1
from src.runtime.memory import MemoryStore
from src.schemas import AgentBrief, ToolResult
from src.runtime.tracing import TraceLogger
from helpers import FakeLLM, synthetic_frame

def _seed_store_with_tool_data(store, frame):
    """Emulate what Agent A's loop stored after its two successful tools."""
    import numpy as np

    close = frame["Close"].astype(float)
    state = {"sma50_stance": "above", "sma200_stance": "above", "rsi": 61.2,
             "macd_hist": 0.8, "macd_fresh_cross": False, "pct_b": 0.83}
    price_payload = {
        "ticker": "NVDA", "period": "1y",
        "current_price": round(float(close.iloc[-1]), 2),
        "previous_close": round(float(close.iloc[-2]), 2),
        "week52_high": round(float(frame.tail(252)["High"].max()), 2),
        "week52_low": round(float(frame.tail(252)["Low"].min()), 2),
        "ytd_pct": 12.34, "rows": [], "indicator_state": state,
    }
    log_returns = np.log(close / close.shift(1)).dropna().tail(90)
    vol_payload = {
        "ticker": "NVDA", "annualized": round(float(log_returns.std(ddof=0) * np.sqrt(252)), 4),
        "daily": round(float(log_returns.std(ddof=0)), 6), "window": 90,
        "n_obs": 90, "band": "moderate",
    }
    store.add_tool_result("agent_a", config.TOOL_GET_PRICE_DATA,
                          {"ticker": "NVDA", "period": "1y"},
                          ToolResult(ok=True, data=price_payload, source="yfinance").model_dump())
    store.add_tool_result("agent_a", config.TOOL_CALCULATE_VOLATILITY,
                          {"ticker": "NVDA", "window": 90},
                          ToolResult(ok=True, data=vol_payload, source="yfinance").model_dump())

def test_brief_round_trip_is_lossless(tmp_path):
    store = MemoryStore()
    _seed_store_with_tool_data(store, synthetic_frame())
    brief = build_brief_v1("NVDA", store, analyst_notes="trend intact")
    restored = AgentBrief.model_validate_json(brief.model_dump_json())
    assert restored == brief  # lossless typed handoff

def test_brief_numbers_come_from_tool_data_not_the_llm(tmp_path):
    store = MemoryStore()
    frame = synthetic_frame()
    _seed_store_with_tool_data(store, frame)
    llm = FakeLLM([])  # the LLM is never consulted for the numbers
    brief = build_brief_v1("NVDA", store)
    assert brief.price_level.current == pytest.approx(float(frame["Close"].iloc[-1]), abs=0.01)
    assert brief.sentiment is None           # brief v1 has no sentiment by construction
    assert brief.volatility.annualized_90d is not None
    assert brief.indicator_state.sma50_stance == "above"
