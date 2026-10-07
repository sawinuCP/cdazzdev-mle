# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'runtime whitelist enforcement: every cross-access pair refused inside the dispatcher', Date: 2026-10-06
"""Whitelist enforcement: every cross-access pair must be refused."""
from __future__ import annotations

import pytest

from src import config
from src.agents.agents import AgentLoop, CrossAgentToolAccessError
from src.runtime.memory import MemoryStore
from src.runtime.tracing import TraceLogger
from helpers import FakeLLM


def _action(tool: str) -> dict:
    return {"thought": "t", "action": tool, "args": {}, "replan_reason": None}


CROSS_PAIRS = [
    ("agent_a", config.TOOL_GET_NEWS),
    ("agent_a", config.TOOL_WEB_SEARCH),
    ("agent_b", config.TOOL_GET_PRICE_DATA),
    ("agent_b", config.TOOL_CALCULATE_VOLATILITY),
    ("agent_b", config.TOOL_LLM_SENTIMENT),
]


@pytest.mark.parametrize("role,forbidden_tool", CROSS_PAIRS)
def test_cross_access_refused_at_dispatcher(tmp_path, role, forbidden_tool):
    trace = TraceLogger(path=tmp_path / "trace.jsonl")
    store = MemoryStore()
    llm = FakeLLM([_action(forbidden_tool), _action("finish")])
    loop = AgentLoop(role, "NVDA", config.WHITELISTS[role], llm, trace, store,
                     config.ROLE_MAX_STEPS, "objective")
    summary = loop.run()
    refused = [o for o in summary["observations"] if "REFUSED" in o["digest"]]
    assert refused, f"{role} -> {forbidden_tool} was not refused"
    assert forbidden_tool in refused[0]["digest"] or "not permitted" in refused[0]["digest"]


def test_dispatcher_raises_cross_agent_error(tmp_path):
    from src.schemas import AgentAction

    trace = TraceLogger(path=tmp_path / "trace.jsonl")
    store = MemoryStore()
    llm = FakeLLM([])
    loop = AgentLoop("agent_b", "NVDA", config.WHITELISTS["agent_b"], llm, trace,
                     store, 2, "objective")
    action = AgentAction(thought="t", action=config.TOOL_GET_PRICE_DATA,
                         args={"ticker": "NVDA", "period": "1y"})
    with pytest.raises(CrossAgentToolAccessError) as excinfo:
        loop._dispatch(action)
    assert "agent_b" in str(excinfo.value) and "get_price_data" in str(excinfo.value)
