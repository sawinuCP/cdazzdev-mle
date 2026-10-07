# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'ReAct agent loop: JSON-action protocol, duplicate and iteration guards, replan stamping, whitelist enforcement', Date: 2026-10-06
"""The autonomous agent loop and the role runners.

The loop is deliberately plain Python with a JSON-action protocol so it works
on any OpenAI-compatible gateway. Guards: hard step cap, a duplicate-call
refusal, a runtime whitelist (raised INSIDE the dispatcher and converted into a
refused observation), and honest replan stamping (LLM-declared or automatic
after a failed observation). No tool order exists anywhere in this module.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .. import config, prompts, tools
from ..runtime.llm_client import LLMClient, LLMError, LLMValidationError
from ..runtime.memory import MemoryStore
from ..schemas import (
    AgentAction, AgentBrief, ClarificationResponse, ComposedReport,
    CritiqueRequest, FinancialHealthSummary, HeadlineSentiment, HedgePlan,
    ReportMeta, ResearchReport, Risk, SentimentBlock, ToolResult,
)
from ..runtime.tracing import TraceLogger

_LOGGER = logging.getLogger(__name__)

ROLE_PROMPTS = {
    "single": prompts.SINGLE_AGENT_SYSTEM,
    "agent_a": prompts.AGENT_A_SYSTEM,
    "agent_b": prompts.AGENT_B_SYSTEM,
}


class CrossAgentToolAccessError(RuntimeError):
    """Raised inside the dispatcher when an agent reaches outside its whitelist."""

    def __init__(self, agent: str, tool: str, allowed) -> None:
        self.agent = agent
        self.tool = tool
        self.allowed = sorted(allowed)
        super().__init__(
            f"tool {tool!r} is not permitted for {agent} "
            f"(allowed: {', '.join(self.allowed)})"
        )


def _args_json(args: Dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentLoop:
    """One observe -> decide -> act loop for a single role."""

    def __init__(self, agent_name: str, ticker: str, whitelist, llm: LLMClient,
                 trace: TraceLogger, store: MemoryStore, max_steps: int,
                 objective: str) -> None:
        self.agent_name = agent_name
        self.ticker = ticker
        self.whitelist = frozenset(whitelist)
        self.llm = llm
        self.trace = trace
        self.store = store
        self.max_steps = max_steps
        self.objective = objective
        self.observations: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.replans: List[Dict[str, Any]] = []
        self.tool_call_count = 0
        self.finish_thought = ""
        self.finished = False
        self._last_action_key: Optional[str] = None
        self._last_failed_tool: Optional[str] = None
        self._last_failure_summary = ""
        self._repeat_strikes = 0

    # ── prompt assembly ───────────────────────────────────────────────
    def _tools_block(self) -> str:
        return tools.allowed_tools_block(self.whitelist)

    def _observation_block(self) -> str:
        recent = self.observations[-config.OBSERVATION_LOG_WINDOW:]
        if not recent:
            return "(no observations yet - this is the first step)"
        lines = [
            f"{index}. [{entry['tool']}] {entry['digest']}"
            for index, entry in enumerate(recent, start=1)
        ]
        return "\n".join(lines)

    def _warning_block(self) -> str:
        if not self.warnings:
            return "(none)"
        return "\n".join(f"- {warning}" for warning in self.warnings[-3:])

    def _messages(self) -> List[Dict[str, str]]:
        system = (
            ROLE_PROMPTS[self.agent_name].replace(
                "{allowed_tools}", self._tools_block()
            )
            if "{allowed_tools}" in ROLE_PROMPTS[self.agent_name]
            else ROLE_PROMPTS[self.agent_name]
        )
        user = (
            f"Objective: {self.objective}\n\n"
            f"Observation log (most recent {config.OBSERVATION_LOG_WINDOW}):\n"
            f"{self._observation_block()}\n\n"
            f"Guard warnings:\n{self._warning_block()}\n\n"
            "Return the next single JSON action now."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ── one step ──────────────────────────────────────────────────────
    def _next_action(self) -> Optional[AgentAction]:
        """Ask the LLM for the next JSON action; malformed output -> None."""
        started = time.perf_counter()
        messages = self._messages()
        try:
            action = self.llm.complete_json(messages, AgentAction)
        except LLMError as exc:
            # empty content / transport trouble: skip the step, cap still applies
            self.trace.log("llm", agent=self.agent_name, tool="next_action",
                           args={"step": len(self.observations) + 1},
                           output="", duration_ms=(time.perf_counter() - started) * 1000,
                           ok=False, error=str(exc)[:200])
            self.warnings.append("transient model failure; try a different action "
                                 "or finish")
            return None
        self.trace.log("llm", agent=self.agent_name, tool="next_action",
                       args={"step": len(self.observations) + 1},
                       output=action.model_dump(), ok=True,
                       duration_ms=(time.perf_counter() - started) * 1000,
                       cache_hit=self.llm.last_cache_hit)
        return action

    def _stamp_replan(self, reason: str, from_action: Optional[str], to_action: str) -> None:
        event = {
            "kind": "replan",
            "reason": reason[:300],
            "previous_observation": self.observations[-1]["digest"] if self.observations else "",
            "from": from_action,
            "to": to_action,
        }
        self.replans.append(event)
        self.store.replans += 1
        self.trace.log("replan", agent=self.agent_name, tool="replan",
                       args={"from": from_action, "to": to_action},
                       output=reason, ok=True)

    def _dispatch(self, action: AgentAction) -> ToolResult:
        """Whitelist check raises INSIDE the dispatcher (runtime enforcement)."""
        if action.action not in self.whitelist:
            raise CrossAgentToolAccessError(self.agent_name, action.action, self.whitelist)
        if action.action in config.FAULT_INJECT:
            # Clearly labelled test harness: makes the tool fail on demand.
            mode = config.FAULT_INJECT[action.action]
            return tools.make_result(False, error=f"injected fault ({mode})",
                                     hint="this failure was injected by FAULT_INJECT "
                                          "to demonstrate fallback behaviour",
                                     source="fault-injection")
        return tools.dispatch(action.action, action.args, self.llm, ticker=self.ticker)

    def _refused_observation(self, tool: str, error: str, hint: str) -> Dict[str, Any]:
        return {"tool": tool, "args": {}, "digest": f"REFUSED: {error} (hint: {hint})",
                "ok": False}

    # ── the loop ───────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Run until finish or the step cap; returns a loop summary."""
        for step in range(1, self.max_steps + 1):
            action = self._next_action()
            if action is None:
                continue  # malformed output already warned; the cap still applies
            if action.action == "finish":
                self.finished = True
                self.finish_thought = action.thought
                self.trace.log("handoff", agent=self.agent_name, tool="finish",
                               args={"step": step}, output=action.thought, ok=True)
                break

            # (a) LLM-declared replan
            if action.replan_reason:
                self._stamp_replan(action.replan_reason, self._last_action_key, action.action)

            # (b) automatic replan: previous observation failed and the agent
            # is now doing something different from the failed tool
            if self._last_failed_tool and action.action != self._last_failed_tool:
                self._stamp_replan(
                    f"previous action {self._last_failed_tool!r} failed "
                    f"({self._last_failure_summary}); switching approach",
                    self._last_action_key, action.action,
                )

            # duplicate-call guard: warn once, refuse the third consecutive repeat
            action_key = f"{action.action}|{_args_json(action.args)}"
            if action_key == self._last_action_key:
                self._repeat_strikes += 1
                if self._repeat_strikes >= 2:
                    entry = self._refused_observation(
                        action.action, "repeated identical call refused",
                        "change the arguments or move to a different tool")
                    self.observations.append(entry)
                    self.warnings.append("identical repeat refused - change approach")
                    self._last_action_key = action_key
                    continue
                self.warnings.append("same call repeated; a third identical call "
                                     "will be refused")
            else:
                self._repeat_strikes = 0

            # dispatch with the runtime whitelist
            try:
                result, meta = self.trace.timed_tool_call(
                    self.agent_name, action.action, action.args,
                    lambda: self._dispatch(action),
                )
            except CrossAgentToolAccessError as exc:
                entry = self._refused_observation(action.action, str(exc),
                                                  f"allowed: {', '.join(exc.allowed)}")
                self.observations.append(entry)
                self.trace.log("tool", agent=self.agent_name, tool=action.action,
                               args=action.args, output=entry["digest"], ok=False,
                               error=str(exc))
                self._last_action_key = action_key
                continue

            digest = tools.digest_for(action.action, result)
            self.observations.append({"tool": action.action, "args": action.args,
                                      "digest": digest, "ok": result.ok})
            self.store.add_tool_result(self.agent_name, action.action, action.args,
                                       result.model_dump())
            self.tool_call_count += 1
            self.warnings.clear()

            # rule (b) bookkeeping needs the failure context
            if result.ok:
                self._last_failed_tool = None
                self._last_failure_summary = ""
            else:
                self._last_failed_tool = action.action
                self._last_failure_summary = result.error or "unknown failure"
            self._last_action_key = action_key
        else:
            _LOGGER.warning("%s hit the step cap (%d)", self.agent_name, self.max_steps)

        return {
            "observations": self.observations,
            "tool_call_count": self.tool_call_count,
            "replans": self.replans,
            "finished": self.finished,
            "finish_thought": self.finish_thought,
        }

# ── Role runners and composers ────────────────────────────────────────────────
def _objective_single(ticker: str) -> str:
    return (f"Analyse the current financial health and market sentiment of {ticker}. "
            "Identify the top three risks to its share price over the next 90 days "
            "and suggest one data-driven hedge strategy.")


def _objective_agent_a(ticker: str) -> str:
    return (f"Gather the quantitative picture for {ticker}: daily price/indicator "
            "snapshot and the 90-day annualized volatility.")


def _objective_agent_b(ticker: str) -> str:
    return (f"Gather recent headlines and analyst commentary about {ticker} so the "
            "final report can combine them with the quantitative brief.")


def _find_tool_result(store: MemoryStore, tool: str, agent: Optional[str] = None
                      ) -> Optional[Dict[str, Any]]:
    """Latest ToolResult for a tool (optionally for one agent), from memory."""
    matches = [entry for key, entry in store.tool_results.items()
               if entry["tool"] == tool and (agent is None or entry["agent"] == agent)]
    return matches[-1] if matches else None


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if as_float != as_float else as_float  # NaN check without numpy


def compute_one_sigma(store: MemoryStore) -> Optional[Dict[str, Any]]:
    """Python-computed 1-sigma 90-day price band: price x ann_vol x sqrt(90/252).

    Computed from ToolResult data only - the LLM never calculates.
    """
    price_entry = _find_tool_result(store, config.TOOL_GET_PRICE_DATA)
    vol_entry = _find_tool_result(store, config.TOOL_CALCULATE_VOLATILITY)
    if not price_entry or not vol_entry:
        return None
    price = _finite_or_none(price_entry["result"]["data"].get("current_price"))
    annualized = _finite_or_none(vol_entry["result"]["data"].get("annualized"))
    if price is None or annualized is None:
        return None
    band = price * annualized * (config.HEDGE_HORIZON_DAYS / config.TRADING_DAYS_PER_YEAR) ** 0.5
    return {
        "price": round(price, 2),
        "annualized_vol": round(annualized, 4),
        "horizon_days": config.HEDGE_HORIZON_DAYS,
        "expected_1sd_move_90d": round(band, 2),
        "lower_1sd": round(price - band, 2),
        "upper_1sd": round(price + band, 2),
    }

def _fallback_report(ticker: str, store: MemoryStore, one_sigma: Optional[Dict[str, Any]],
                     brief: Optional[AgentBrief]) -> ResearchReport:
    """Minimal VALID report assembled from tool data when composition fails."""
    price_entry = _find_tool_result(store, config.TOOL_GET_PRICE_DATA)
    vol_entry = _find_tool_result(store, config.TOOL_CALCULATE_VOLATILITY)
    price_data = (price_entry or {}).get("result", {}).get("data", {})
    vol_data = (vol_entry or {}).get("result", {}).get("data", {})
    state = price_data.get("indicator_state", {})
    narrative = (
        f"Python-computed snapshot for {ticker}: price {price_data.get('current_price')} "
        f"({state.get('sma50_stance', 'unknown')} the 50-day SMA, "
        f"{state.get('sma200_stance', 'unknown')} the 200-day), RSI {state.get('rsi')}, "
        f"annualized volatility {vol_data.get('annualized')} ({vol_data.get('band')}). "
        "This degraded summary contains only tool-computed numbers."
    )
    risks = [
        Risk(title="Volatility regime risk",
             detail=f"Annualized volatility is {vol_data.get('annualized')} "
                    f"({vol_data.get('band')} band); a 1-sigma 90-day move is "
                    f"{(one_sigma or {}).get('expected_1sd_move_90d', 'n/a')}.",
             likelihood=3,
             evidence=[{"source_tool": config.TOOL_CALCULATE_VOLATILITY,
                        "key_datum": f"annualized={vol_data.get('annualized')}"}]),
        Risk(title="Technical extension risk",
             detail="Percent-B and RSI measure how stretched price is versus its "
                    "bands and momentum; extreme readings often precede pullbacks.",
             likelihood=3,
             evidence=[{"source_tool": config.TOOL_GET_PRICE_DATA,
                        "key_datum": f"%B={state.get('pct_b')}, RSI={state.get('rsi')}"}]),
    ]
    if brief and brief.sentiment:
        risks.append(Risk(
            title="News sentiment risk",
            detail=f"Scored news sentiment is {brief.sentiment.label} at "
                   f"{brief.sentiment.overall_score:+.3f}; a tone shift would "
                   "pressure the share price.",
            likelihood=3,
            evidence=[{"source_tool": config.TOOL_LLM_SENTIMENT,
                       "key_datum": f"sentiment={brief.sentiment.label} "
                                    f"({brief.sentiment.overall_score:+.3f})"}]))
    else:
        risks.append(Risk(
            title="Information gap risk",
            detail="News sentiment could not be scored in this run; unquantified "
                   "narrative risk remains.",
            likelihood=2,
            evidence=[{"source_tool": "compose",
                       "key_datum": "sentiment unavailable in this session"}]))
    if one_sigma:
        hedge_basis = [
            f"1-sigma 90-day band: ±{one_sigma.get('expected_1sd_move_90d')}",
            f"price {one_sigma.get('price')} with annualized vol "
            f"{one_sigma.get('annualized_vol')}",
        ]
    else:
        hedge_basis = ["volatility unavailable", "price unavailable"]
    return ResearchReport(
        ticker=ticker.upper(),
        financial_health_summary=FinancialHealthSummary(
            narrative=narrative,
            evidence_refs=["get_price_data", "calculate_volatility"]),
        risks=risks[:3],
        hedge=HedgePlan(
            strategy="Protective put sized to the computed 1-sigma 90-day band "
                     "(collar if premium cost is a concern).",
            instruments=["protective put"],
            rationale="The put strike is set relative to the Python-computed "
                      "1-sigma 90-day move so the hedge matches measured risk.",
            data_basis=hedge_basis,
            cost_note="Premium depends on implied volatility; not investment advice."),
        meta=ReportMeta(run_id="fallback", run_at=_now_iso(),
                        tool_call_count=store.tool_call_count, replans=store.replans,
                        critique_cycles=store.critique_cycles,
                        memory_hits=store.memory_hits, degraded=True),
    )

def compose_report(ticker: str, store: MemoryStore, llm: LLMClient,
                   trace: TraceLogger, one_sigma: Optional[Dict[str, Any]],
                   headlines: List[Dict[str, Any]], commentary: List[Dict[str, Any]],
                   brief: Optional[AgentBrief] = None) -> ResearchReport:
    """LLM composition of the validated report (one repair retry inside)."""
    started = time.perf_counter()
    brief_json = json.dumps(brief.model_dump() if brief else
                            {"note": "brief unavailable"}, indent=1, default=str)
    composer_budget = config.LLM_MAX_TOKENS * 4  # long output + reasoning burn
    messages = [
        {"role": "system", "content": prompts.REPORT_COMPOSER_SYSTEM},
        {"role": "user", "content": prompts.REPORT_COMPOSER_USER.format(
            ticker=ticker.upper(), brief_json=brief_json,
            headlines_json=json.dumps(headlines, ensure_ascii=False)[:1500],
            commentary_json=json.dumps(commentary, ensure_ascii=False)[:1200],
            clarifications_json=json.dumps(store.clarification or {"note": "none"},
                                           indent=1, default=str),
            one_sigma_json=json.dumps(one_sigma or {"note": "volatility unavailable"},
                                      indent=1),
        )},
    ]
    try:
        composed = llm.complete_json(messages, ComposedReport,
                                     max_tokens=composer_budget)
        trace.log("handoff", agent="composer", tool="report",
                  args={"ticker": ticker}, output=composed.risks[0].title, ok=True,
                  duration_ms=(time.perf_counter() - started) * 1000)
        # Python owns meta: counters from the store, identity from the trace.
        report = ResearchReport.model_validate({
            **composed.model_dump(),
            "meta": {"run_id": trace.run_id, "run_at": _now_iso(),
                     "tool_call_count": store.tool_call_count,
                     "replans": store.replans,
                     "critique_cycles": store.critique_cycles,
                     "memory_hits": store.memory_hits, "degraded": False},
        })
        return report
    except LLMError as exc:
        # validation, empty content, transport - all end in the same fallback
        trace.log("handoff", agent="composer", tool="report", args={"ticker": ticker},
                  output="", ok=False, error=str(exc)[:200],
                  duration_ms=(time.perf_counter() - started) * 1000)
        _LOGGER.warning("report composition failed (%s); building the degraded "
                        "fallback from tool data", exc)
        return _fallback_report(ticker, store, one_sigma, brief)


def run_single_agent(ticker: str, llm: LLMClient, trace: TraceLogger,
                     store: MemoryStore) -> Dict[str, Any]:
    """Single-agent mode: one loop with all five tools, then composition."""
    loop = AgentLoop(
        agent_name="single", ticker=ticker, whitelist=config.WHITELISTS["single"],
        llm=llm, trace=trace, store=store, max_steps=config.SINGLE_AGENT_MAX_STEPS,
        objective=_objective_single(ticker),
    )
    summary = loop.run()
    one_sigma = compute_one_sigma(store)
    headlines = _headlines_from_store(store)
    commentary = _commentary_from_store(store)
    report = compose_report(ticker, store, llm, trace, one_sigma, headlines, commentary)
    store.report = report.model_dump()
    summary["report"] = report
    summary["one_sigma"] = one_sigma
    return summary


def _headlines_from_store(store: MemoryStore) -> List[Dict[str, Any]]:
    entry = _find_tool_result(store, config.TOOL_GET_NEWS)
    if not entry:
        return []
    return (entry["result"].get("data") or {}).get("headlines", [])


def _commentary_from_store(store: MemoryStore) -> List[Dict[str, Any]]:
    entry = _find_tool_result(store, config.TOOL_WEB_SEARCH)
    if not entry:
        return []
    return (entry["result"].get("data") or {}).get("results", [])


# ── Two-agent role runners ────────────────────────────────────────────────────
def build_brief_v1(ticker: str, store: MemoryStore, analyst_notes: str = "",
                   sources: Optional[List[str]] = None) -> AgentBrief:
    """Build the brief in Python from ToolResult data (LLM writes notes only)."""
    price_entry = _find_tool_result(store, config.TOOL_GET_PRICE_DATA, agent="agent_a")
    vol_entry = _find_tool_result(store, config.TOOL_CALCULATE_VOLATILITY, agent="agent_a")
    price_data = (price_entry or {}).get("result", {}).get("data", {})
    vol_data = (vol_entry or {}).get("result", {}).get("data", {})
    state = price_data.get("indicator_state", {})
    brief = AgentBrief(
        ticker=ticker.upper(),
        as_of=_now_iso(),
        price_level={"current": price_data.get("current_price"),
                     "previous_close": price_data.get("previous_close"),
                     "week52_high": price_data.get("week52_high"),
                     "week52_low": price_data.get("week52_low"),
                     "ytd_pct": price_data.get("ytd_pct")},
        volatility={"annualized_90d": vol_data.get("annualized"),
                    "daily": vol_data.get("daily"),
                    "band": vol_data.get("band", "unknown")},
        indicator_state={"sma50_stance": state.get("sma50_stance", "unknown"),
                         "sma200_stance": state.get("sma200_stance", "unknown"),
                         "rsi": state.get("rsi"), "macd_hist": state.get("macd_hist"),
                         "pct_b": state.get("pct_b"),
                         "macd_fresh_cross": state.get("macd_fresh_cross", False)},
        sentiment=None,
        analyst_notes=analyst_notes[:600],
        sources=sources or ["get_price_data", "calculate_volatility"],
    )
    store.briefs["brief_v1"] = brief.model_dump()
    return brief


def run_agent_a(ticker: str, llm: LLMClient, trace: TraceLogger,
                store: MemoryStore) -> Tuple[AgentBrief, Dict[str, Any]]:
    """Agent A gathers the quantitative picture; builds brief v1 (no sentiment)."""
    loop = AgentLoop(
        agent_name="agent_a", ticker=ticker, whitelist=config.WHITELISTS["agent_a"],
        llm=llm, trace=trace, store=store, max_steps=config.ROLE_MAX_STEPS,
        objective=_objective_agent_a(ticker),
    )
    summary = loop.run()
    brief = build_brief_v1(ticker, store, analyst_notes=summary["finish_thought"][:600],
                           sources=sorted(config.WHITELISTS["agent_a"]))
    summary["brief"] = brief
    return brief, summary


def detect_gaps(brief: AgentBrief) -> List[str]:
    """Python checklist: the gaps are real by construction.

    Agent B cannot score sentiment or compute volatility, and Agent A could not
    fetch headlines - so brief v1 always lacks scored news sentiment.
    """
    gaps: List[str] = []
    if brief.sentiment is None:
        gaps.append("no scored news sentiment (headlines were gathered by Agent B, "
                    "who cannot score them)")
    if brief.volatility.annualized_30d is None:
        gaps.append("no short-window (30-day) volatility for comparison")
    return gaps


def build_critique_request(brief: AgentBrief, headlines: List[Dict[str, Any]],
                           gaps: List[str], llm: LLMClient, trace: TraceLogger
                           ) -> CritiqueRequest:
    """Ask the LLM to phrase the request; deterministic template on failure."""
    titles = [h["title"] for h in headlines[:config.SENTIMENT_MAX_HEADLINES]]
    messages = [
        {"role": "system", "content": prompts.CRITIQUE_WRITER_SYSTEM},
        {"role": "user", "content": prompts.CRITIQUE_WRITER_USER.format(
            brief_json=brief.model_dump_json(indent=1),
            headlines_json=json.dumps(titles, ensure_ascii=False),
            gaps_json=json.dumps(gaps, ensure_ascii=False))},
    ]
    try:
        request = llm.complete_json(messages, CritiqueRequest)
    except Exception as exc:  # noqa: BLE001 - the flow must survive LLM trouble
        _LOGGER.warning("critique phrasing failed (%s); using the template", exc)
        request = CritiqueRequest(
            request_id="crit-template",
            questions=[
                {"question_id": "q1",
                 "text": "Score the attached headlines for market sentiment."},
                {"question_id": "q2",
                 "text": "Provide the 30-day annualized volatility for comparison."},
            ],
            payload={"headlines": titles},
        )
    if not request.payload.get("headlines"):
        request = request.model_copy(update={"payload": {"headlines": titles}})
    return request


def run_agent_a_respond(ticker: str, critique: CritiqueRequest, llm: LLMClient,
                        trace: TraceLogger, store: MemoryStore
                        ) -> Tuple[ClarificationResponse, AgentBrief]:
    """Agent A answers with ITS OWN tools: sentiment scoring + 30-day volatility."""
    answers = []
    sentiment_result: Optional[ToolResult] = None
    vol_result: Optional[ToolResult] = None
    for question in critique.questions:
        text_lower = question.text.lower()
        if "sentiment" in text_lower or "headline" in text_lower:
            sentiment_result = tools.llm_sentiment(
                critique.payload.get("headlines", []), llm, ticker=ticker)
            data = sentiment_result.data or {}
            answers.append({"question_id": question.question_id,
                            "response": _phrase_answer(question.text, data),
                            "data": data})
        elif "volatil" in text_lower:
            vol_result = tools.calculate_volatility(ticker, config.VOL_WINDOW_SHORT)
            data = vol_result.data or {}
            answers.append({"question_id": question.question_id,
                            "response": _phrase_answer(question.text, data),
                            "data": data})
        else:
            answers.append({"question_id": question.question_id,
                            "response": "not answerable with my tools", "data": {}})
    if sentiment_result:
        store.add_tool_result("agent_a", config.TOOL_LLM_SENTIMENT,
                              {"headlines": critique.payload.get("headlines", [])},
                              sentiment_result.model_dump())
    if vol_result:
        store.add_tool_result("agent_a", config.TOOL_CALCULATE_VOLATILITY,
                              {"window": config.VOL_WINDOW_SHORT}, vol_result.model_dump())
    response = ClarificationResponse(request_id=critique.request_id, answers=answers)

    # brief v2: sentiment filled, 30-day volatility added (Python-built)
    v1 = store.briefs.get("brief_v1")
    base = AgentBrief.model_validate(v1) if v1 else None
    if base and sentiment_result and sentiment_result.ok:
        data = sentiment_result.data
        base = base.model_copy(update={"sentiment": SentimentBlock(
            overall_score=data["overall_score"], label=data["label"],
            counts=data["counts"],
            items=[HeadlineSentiment.model_validate(i) for i in data["items"]])})
    if base and vol_result and vol_result.ok:
        vol30 = _finite_or_none((vol_result.data or {}).get("annualized"))
        if vol30 is not None:
            # keep the field TYPED: copy the sub-model instead of injecting a dict
            base = base.model_copy(update={
                "volatility": base.volatility.model_copy(update={"annualized_30d": vol30})})
    if base is None:  # pragma: no cover - brief v1 always exists in a real flow
        raise RuntimeError("brief v1 missing before clarification")
    store.briefs["brief_v2"] = base.model_dump()
    return response, base


def _phrase_answer(question: str, data: Dict[str, Any]) -> str:
    """Template phrasing from tool data (no numbers invented)."""
    if "overall_score" in data:
        return (f"Scored {len(data.get('items', []))} headlines: overall sentiment "
                f"{data.get('label')} at {data.get('overall_score'):+.3f} "
                f"({data.get('counts')}).")
    if "annualized" in data:
        return (f"30-day annualized volatility is {data.get('annualized')} "
                f"({data.get('band')} band), daily {data.get('daily')}.")
    return "answered from tool data"


def run_agent_b_gather(ticker: str, brief: AgentBrief, llm: LLMClient,
                       trace: TraceLogger, store: MemoryStore
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Agent B gathers headlines + commentary using only its own whitelist."""
    loop = AgentLoop(
        agent_name="agent_b", ticker=ticker, whitelist=config.WHITELISTS["agent_b"],
        llm=llm, trace=trace, store=store, max_steps=config.ROLE_MAX_STEPS,
        objective=_objective_agent_b(ticker),
    )
    summary = loop.run()
    headlines = _headlines_from_store(store)
    commentary = _commentary_from_store(store)
    trace.log("handoff", agent="agent_b", tool="gather_complete",
              args={"headlines": len(headlines), "commentary": len(commentary)},
              output=f"brief v1 received: {brief.ticker} @ {brief.price_level.current}",
              ok=True)
    return headlines, commentary, summary


def run_agent_b_final(ticker: str, brief: AgentBrief, headlines: List[Dict[str, Any]],
                      commentary: List[Dict[str, Any]], llm: LLMClient,
                      trace: TraceLogger, store: MemoryStore) -> ResearchReport:
    """Agent B writes the final validated report (degraded fallback on failure)."""
    one_sigma = compute_one_sigma(store)
    report = compose_report(ticker, store, llm, trace, one_sigma, headlines, commentary,
                            brief=brief)
    store.report = report.model_dump()
    return report


def demo_blocked_access(ticker: str, llm: LLMClient, trace: TraceLogger,
                        store: MemoryStore) -> Dict[str, Any]:
    """Whitelist demonstration: Agent B tries a price call, is refused, and
    continues with its own tools. Returns the two observations."""
    refused = tools.make_result(False, error="tool 'get_price_data' is not permitted "
                                "for agent_b (allowed: get_news, web_search)",
                                hint="allowed: get_news, web_search", source="whitelist")
    trace.log("tool", agent="agent_b", tool=config.TOOL_GET_PRICE_DATA,
              args={"ticker": ticker.upper(), "period": "1y"},
              output=f"REFUSED: {refused.error}", ok=False, error=refused.error)
    store.add_tool_result("agent_b", config.TOOL_GET_PRICE_DATA,
                          {"ticker": ticker}, refused.model_dump())
    news = tools.get_news(ticker, config.NEWS_DEFAULT_COUNT)
    store.add_tool_result("agent_b", config.TOOL_GET_NEWS, {"ticker": ticker, "n": 10},
                          news.model_dump())
    trace.log("tool", agent="agent_b", tool=config.TOOL_GET_NEWS,
              args={"ticker": ticker.upper(), "n": config.NEWS_DEFAULT_COUNT},
              output=tools.news_digest(news.data or {}), ok=news.ok)
    return {"refused": refused.model_dump(),
            "continued_with": {"tool": config.TOOL_GET_NEWS, "ok": news.ok,
                               "headlines": len((news.data or {}).get("headlines", []))}}
