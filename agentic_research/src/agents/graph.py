"""LangGraph state machine for the two-agent research mode.

The typed state flows through: cache_check -> agent_a_brief ->
agent_b_gather_and_critique -> (conditional) agent_a_respond ->
agent_b_final_report -> save_cache. A crash wrapper guarantees no node can
escape with an unhandled exception: failures return ``degraded=True``.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .. import config
from ..runtime import memory as memory_mod
from .agents import (
    AgentBrief, _fallback_report, build_critique_request, compute_one_sigma,
    detect_gaps, run_agent_a, run_agent_a_respond, run_agent_b_final,
    run_agent_b_gather,
)
from ..runtime.llm_client import LLMClient
from ..runtime.memory import MemoryStore, load_cache, save_cache
from ..schemas import CritiqueRequest
from ..runtime.tracing import TraceLogger

_LOGGER = logging.getLogger(__name__)

class ResearchState(TypedDict, total=False):
    """Typed graph state (values may hold non-serializable runtime objects)."""

    ticker: str
    run_id: str
    llm: Any
    trace: Any
    store: Any
    cache_hit: bool
    cache_payload: Dict[str, Any]
    brief: Dict[str, Any]
    critique: Dict[str, Any]
    headlines: List[Dict[str, Any]]
    commentary: List[Dict[str, Any]]
    gaps: List[str]
    clarification: Dict[str, Any]
    report: Dict[str, Any]
    critique_count: int
    tool_call_count: int
    replans: int
    degraded: bool
    error: str

def _crash_wrapper(node_fn):
    """No node may escape with an unhandled exception."""

    def wrapped(state: ResearchState) -> Dict[str, Any]:
        try:
            return node_fn(state)
        except Exception as exc:  # noqa: BLE001 - the wrapper is the safety net
            _LOGGER.exception("node %s crashed: %s", node_fn.__name__, exc)
            return {"degraded": True,
                    "error": f"{node_fn.__name__}: {type(exc).__name__}: {exc}"}

    wrapped.__name__ = node_fn.__name__
    return wrapped

def _node_cache_check(state: ResearchState) -> Dict[str, Any]:
    payload, error = load_cache(state["ticker"], config.CACHE_DIR)
    if error:
        state["trace"].log("cache", agent="system", tool="load_cache",
                           args={"ticker": state["ticker"]}, output=error, ok=False)
        return {"cache_hit": False}
    if payload:
        state["trace"].log("cache", agent="system", tool="load_cache",
                           args={"ticker": state["ticker"]},
                           output=f"loaded brief for {state['ticker']} on "
                                  f"{payload.get('as_of', '')}", ok=True)
        return {"cache_hit": True, "cache_payload": payload}
    state["trace"].log("cache", agent="system", tool="load_cache",
                       args={"ticker": state["ticker"]},
                       output="no cache for today - full run", ok=True)
    return {"cache_hit": False}

def _node_agent_a_brief(state: ResearchState) -> Dict[str, Any]:
    brief, _summary = run_agent_a(state["ticker"], state["llm"], state["trace"],
                                  state["store"])
    state["trace"].log("handoff", agent="agent_a", tool="agent_brief_v1",
                       args={"ticker": state["ticker"]},
                       output=brief.model_dump_json(), ok=True)
    return {"brief": brief.model_dump(),
            "tool_call_count": state["store"].tool_call_count,
            "replans": state["store"].replans}

def _node_agent_b_gather_and_critique(state: ResearchState) -> Dict[str, Any]:
    brief = AgentBrief.model_validate(state["brief"])
    headlines, commentary, _gather = run_agent_b_gather(
        state["ticker"], brief, state["llm"], state["trace"], state["store"])
    gaps = detect_gaps(brief)
    critique = build_critique_request(brief, headlines, gaps, state["llm"],
                                      state["trace"])
    state["trace"].log("critique", agent="agent_b", tool="critique_request",
                       args={"gaps": gaps},
                       output=critique.model_dump_json(), ok=True)
    # critique_count stays 0 until the clarification actually runs; the routing
    # check (critique_count < cap) must see the cycle as still available here.
    return {"critique": critique.model_dump(), "headlines": headlines,
            "commentary": commentary, "gaps": gaps, "critique_count": 0}

def _node_agent_a_respond(state: ResearchState) -> Dict[str, Any]:
    critique = CritiqueRequest.model_validate(state["critique"])
    response, brief_v2 = run_agent_a_respond(state["ticker"], critique, state["llm"],
                                             state["trace"], state["store"])
    state["trace"].log("handoff", agent="agent_a", tool="clarification_response",
                       args={"request_id": response.request_id},
                       output=f"{len(response.answers)} answers", ok=True)
    state["store"].critique_cycles += 1  # the cycle landed: counted once per graph
    return {"clarification": response.model_dump(), "brief": brief_v2.model_dump(),
            "tool_call_count": state["store"].tool_call_count,
            "replans": state["store"].replans, "critique_count": 1}

def _node_agent_b_final_report(state: ResearchState) -> Dict[str, Any]:
    brief = AgentBrief.model_validate(state["brief"])
    report = run_agent_b_final(state["ticker"], brief,
                               state.get("headlines", []),
                               state.get("commentary", []),
                               state["llm"], state["trace"], state["store"])
    return {"report": report.model_dump(),
            "tool_call_count": state["store"].tool_call_count,
            "replans": state["store"].replans}

def _node_save_cache(state: ResearchState) -> Dict[str, Any]:
    payload = {
        "as_of": date.today().isoformat(),
        "ticker": state["ticker"],
        "brief_v2": state.get("brief"),
        "clarification": state.get("clarification"),
        "report": state.get("report") or {},
        "meta": {"run_id": state["run_id"],
                 "tool_call_count": state.get("tool_call_count", 0),
                 "replans": state.get("replans", 0),
                 "critique_cycles": state.get("critique_count", 0)},
    }
    save_cache(payload, state["ticker"], config.CACHE_DIR)
    return {}

def _route_cache(state: ResearchState) -> str:
    """Conditional edge after the cache probe."""
    return "hit" if state.get("cache_hit") else "miss"

def _node_degraded_report(state: ResearchState) -> Dict[str, Any]:
    """Terminal path when a node crashed: minimal valid report from tool data."""
    report = _fallback_report(state["ticker"], state["store"],
                              compute_one_sigma(state["store"]), None)
    state["trace"].log("handoff", agent="system", tool="degraded_report",
                       args={"ticker": state["ticker"]},
                       output=state.get("error", "degraded"), ok=False)
    return {"report": report.model_dump(), "degraded": True}

def build_graph(llm: LLMClient, trace: TraceLogger, store: MemoryStore):
    """Compile the research graph with the shared runtime objects bound in.

    The runtime objects (client, trace, store) are injected through the state
    dict at invoke time, so tests can substitute mocks the same way. Any node
    that crashes routes to the degraded terminal node instead of cascading.
    """

    def cache_check(state: ResearchState) -> Dict[str, Any]:
        return _node_cache_check(state)

    def agent_a_brief(state: ResearchState) -> Dict[str, Any]:
        return _crash_wrapper(_node_agent_a_brief)(state)

    def agent_b_gather(state: ResearchState) -> Dict[str, Any]:
        return _crash_wrapper(_node_agent_b_gather_and_critique)(state)

    def agent_a_respond(state: ResearchState) -> Dict[str, Any]:
        return _crash_wrapper(_node_agent_a_respond)(state)

    def agent_b_final(state: ResearchState) -> Dict[str, Any]:
        return _crash_wrapper(_node_agent_b_final_report)(state)

    def save_cache_node(state: ResearchState) -> Dict[str, Any]:
        return _node_save_cache(state)

    def degraded_report(state: ResearchState) -> Dict[str, Any]:
        return _node_degraded_report(state)

    def _route_after_a(state: ResearchState) -> str:
        return "degraded" if state.get("degraded") else "gather"

    def _route_after_gather(state: ResearchState) -> str:
        if state.get("degraded"):
            return "degraded"
        if state.get("gaps") and state.get("critique_count", 0) < config.CRITIQUE_CYCLE_CAP:
            return "clarify"
        return "final"

    def _route_after_respond(state: ResearchState) -> str:
        return "degraded" if state.get("degraded") else "final"

    def _route_after_final(state: ResearchState) -> str:
        return "degraded" if state.get("degraded") else "save"

    graph = StateGraph(ResearchState)
    graph.add_node("cache_check", cache_check)
    graph.add_node("agent_a_brief", agent_a_brief)
    graph.add_node("agent_b_gather_and_critique", agent_b_gather)
    graph.add_node("agent_a_respond", agent_a_respond)
    graph.add_node("agent_b_final_report", agent_b_final)
    graph.add_node("save_cache", save_cache_node)
    graph.add_node("degraded_report", degraded_report)
    graph.set_entry_point("cache_check")
    graph.add_conditional_edges("cache_check", _route_cache,
                                {"hit": END, "miss": "agent_a_brief"})
    graph.add_conditional_edges("agent_a_brief", _route_after_a,
                                {"gather": "agent_b_gather_and_critique",
                                 "degraded": "degraded_report"})
    graph.add_conditional_edges("agent_b_gather_and_critique", _route_after_gather,
                                {"clarify": "agent_a_respond",
                                 "final": "agent_b_final_report",
                                 "degraded": "degraded_report"})
    graph.add_conditional_edges("agent_a_respond", _route_after_respond,
                                {"final": "agent_b_final_report",
                                 "degraded": "degraded_report"})
    graph.add_conditional_edges("agent_b_final_report", _route_after_final,
                                {"save": "save_cache", "degraded": "degraded_report"})
    graph.add_edge("save_cache", END)
    graph.add_edge("degraded_report", END)
    return graph.compile()

def run_research(ticker: str, llm: Optional[LLMClient] = None,
                 use_cache: bool = True) -> Dict[str, Any]:
    """One-call entry point: full two-agent research run (no manual input)."""
    from ..runtime.llm_client import client_from_env, load_env

    load_env()
    config.ensure_dirs()
    llm = llm or client_from_env(use_cache=use_cache)
    trace = TraceLogger()
    store = MemoryStore()
    memory_mod.LAST_SESSION = store
    app = build_graph(llm, trace, store)
    initial: ResearchState = {
        "ticker": ticker.upper(), "run_id": trace.run_id, "llm": llm, "trace": trace,
        "store": store, "cache_hit": False, "critique_count": 0,
        "tool_call_count": 0, "replans": 0, "degraded": False,
    }
    final = dict(app.invoke(initial))
    if final.get("cache_hit"):
        memory_mod.LAST_SESSION.report = final.get("cache_payload", {}).get("report")
        return {"ticker": ticker.upper(), "run_id": trace.run_id, "cache_hit": True,
                "cache_payload": final.get("cache_payload"), "report": None,
                "degraded": False, "error": "", "trace": trace, "store": store}
    if not final.get("report"):
        report = _fallback_report(ticker, store, compute_one_sigma(store), None)
        return {"ticker": ticker.upper(), "run_id": trace.run_id, "cache_hit": False,
                "report": report.model_dump(), "degraded": True,
                "error": final.get("error") or "graph produced no report",
                "trace": trace, "store": store}
    return {"ticker": ticker.upper(), "run_id": trace.run_id, "cache_hit": False,
            "report": final["report"], "degraded": final.get("degraded", False),
            "error": final.get("error", ""), "trace": trace, "store": store}
