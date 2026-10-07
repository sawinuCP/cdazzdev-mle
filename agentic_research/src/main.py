"""Public entry points and the CLI.

- ``run_single_agent_mode(ticker)``: one agent with all five tools.
- ``run_research(ticker)``: the full two-agent graph (see graph.py).
- ``answer_from_memory(question)``: follow-up answered from the session store
  WITHOUT any tool call (the tool-call counter must not move).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel

from . import config, prompts, tools
from .runtime import memory as memory_module
from .agents.agents import run_single_agent
from .runtime.llm_client import (LLMConfigurationError, LLMClient, client_from_env, load_env)
from .runtime.memory import MemoryStore
from .schemas import ResearchReport
from .runtime.tracing import TraceLogger

_LOGGER = logging.getLogger(__name__)

class MemoryAnswer(BaseModel):
    """Validated reply of the memory answerer."""

    answer: str

def _session() -> MemoryStore:
    # read the memory module attribute DYNAMICALLY: a by-value import would
    # stay None in any module that imported main before a run set the session
    store = memory_module.LAST_SESSION
    if store is None:
        raise RuntimeError(
            "no research session yet - run run_single_agent_mode(ticker) or "
            "run_research(ticker) first."
        )
    return store

def run_single_agent_mode(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """One agent with all five tools; saves the report artefacts."""
    load_env()
    config.ensure_dirs()
    llm = client_from_env(use_cache=use_cache)
    trace = TraceLogger()
    store = MemoryStore()
    memory_module.LAST_SESSION = store
    summary = run_single_agent(ticker.upper(), llm, trace, store)
    report: ResearchReport = summary["report"]
    config.REPORT_JSON.write_text(json.dumps(report.model_dump(), indent=1,
                                             ensure_ascii=False), encoding="utf-8")
    config.REPORT_MD.write_text(_report_markdown(report), encoding="utf-8")
    trace.log("handoff", agent="system", tool="artifacts",
              args={"ticker": ticker}, output=f"{config.REPORT_JSON.name}, "
                                              f"{config.REPORT_MD.name}", ok=True)
    return {"report": report.model_dump(), "run_id": trace.run_id,
            "trace": trace, "store": store}

def answer_from_memory(question: str, llm: Optional[LLMClient] = None,
                       use_cache: bool = True) -> Dict[str, Any]:
    """Answer a follow-up from the session record ONLY (no tool calls)."""
    load_env()
    store = _session()
    counter_before = store.tool_call_count
    llm = llm or client_from_env(use_cache=use_cache)
    trace = TraceLogger(run_id=f"memory-{counter_before}")
    messages = [
        {"role": "system", "content": prompts.MEMORY_ANSWER_SYSTEM},
        {"role": "user", "content": prompts.MEMORY_ANSWER_USER.format(
            question=question,
            memory_json=json.dumps(store.snapshot(), indent=1, default=str)[:6000])},
    ]
    answer = llm.complete_json(messages, MemoryAnswer).answer
    trace.log("memory", agent="memory", tool="answer_from_memory",
              args={"question": question[:120]}, output=answer, ok=True)
    store.memory_hits += 1
    return {"question": question, "answer": answer,
            "tool_call_count_before": counter_before,
            "tool_call_count_after": store.tool_call_count,
            "counter_unchanged": store.tool_call_count == counter_before,
            "trace": trace}

class MemoryAnswer(BaseModel):
    """Validated reply of the memory answerer."""

    answer: str

def _report_markdown(report: ResearchReport) -> str:
    risks = "\n".join(
        f"{index}. **{risk.title}** (likelihood {risk.likelihood}) - {risk.detail}\n"
        f"   evidence: {'; '.join(f'{e.source_tool}: {e.key_datum}' for e in risk.evidence)}"
        for index, risk in enumerate(report.risks, start=1)
    )
    return (
        f"# {report.ticker} — agentic research report\n\n"
        f"## Financial health summary\n\n{report.financial_health_summary.narrative}\n\n"
        f"Evidence: {', '.join(report.financial_health_summary.evidence_refs)}\n\n"
        f"## Top three 90-day risks\n\n{risks}\n\n"
        f"## Hedge strategy\n\n{report.hedge.strategy}\n\n{report.hedge.rationale}\n\n"
        f"Instruments: {', '.join(report.hedge.instruments)}\n"
        f"Data basis: {'; '.join(report.hedge.data_basis)}\n"
        f"Cost note: {report.hedge.cost_note}\n\n"
        f"---\n\n{report.disclaimer}\n"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic financial research system")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--mode", choices=("two-agent", "single"), default="two-agent")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    load_env()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        if args.mode == "single":
            result = run_single_agent_mode(args.ticker, use_cache=not args.no_cache)
            report = result["report"]
        else:
            from .graph import run_research

            result = run_research(args.ticker, use_cache=not args.no_cache)
            if result.get("cache_hit"):
                payload = result.get("cache_payload") or {}
                print(f"cache HIT — loaded brief for {args.ticker.upper()} on "
                      f"{payload.get('as_of', '')}")
                report = payload.get("report") or {}
            else:
                report = result.get("report") or {}
        print(json.dumps(report, indent=1, ensure_ascii=False, default=str)[:2400])
    except LLMConfigurationError as exc:
        print(f"[config] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - friendly CLI failure
        print(f"[error] {type(exc).__name__}: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
