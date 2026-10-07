# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'streamlit trace dashboard: run filter, per-agent tool histogram, duration bars, success pie, replan table', Date: 2026-10-06
"""Streamlit dashboard over the agent trace JSONL.

Run from this folder:  streamlit run dashboards/trace_dashboard.py
Requires: streamlit, plotly (pip install streamlit plotly).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "agent_trace.jsonl"


def load_entries() -> list:
    if not TRACE_PATH.exists():
        return []
    entries = []
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(__import__("json").loads(line))
    return entries


st.set_page_config(page_title="Agentic Research Trace", layout="wide")
st.title("Agentic Research — trace")

entries = load_entries()
if not entries:
    st.warning("No trace yet. Run `python -m src.main --ticker NVDA` first.")
    st.stop()

run_ids = ["all runs"] + sorted({e["run_id"] for e in entries})
selected = st.sidebar.selectbox("Run", run_ids)
if selected != "all runs":
    entries = [e for e in entries if e["run_id"] == selected]

kinds = Counter(e["kind"] for e in entries)
tool_entries = [e for e in entries if e["kind"] == "tool"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("tool calls", len(tool_entries))
c2.metric("replans", kinds.get("replan", 0))
c3.metric("critiques", kinds.get("critique", 0))
c4.metric("memory events", kinds.get("memory", 0))

st.subheader("Tool calls per agent")
tool_plotly = st.container()
tools_by_agent = Counter((e["agent"], e["tool"]) for e in tool_entries)
st.bar_chart(pd.DataFrame(
    [{"agent": k[0], "tool": k[1], "calls": v} for k, v in tools_by_agent.items()]
).pivot_table(index="agent", columns="tool", values="calls", aggfunc="sum").fillna(0))

st.subheader("Duration of successful tool calls")
durations = [(e["agent"], e["tool"], round(e["duration_ms"], 0))
             for e in tool_entries if e.get("ok") and e.get("duration_ms") is not None]
if durations:
    st.bar_chart(pd.DataFrame(durations, columns=["agent", "tool", "ms"])
                 .set_index(["agent", "tool"])["ms"])
else:
    st.info("No timed tool calls recorded (timestamps use time.perf_counter).")

st.subheader("Outcome split")
pie_rows = [{"status": "ok" if e["ok"] else "failed"} for e in tool_entries]
st.bar_chart(pd.DataFrame(pie_rows).groupby("status").size())

st.subheader("Replans and critique events")
replan_rows = [
    {"kind": e["kind"], "agent": e["agent"], "reason": str(e["output"])[:120],
     "from": (e.get("args") or {}).get("from"), "to": (e.get("args") or {}).get("to")}
    for e in entries if e["kind"] in {"replan", "critique"}
]
st.dataframe(pd.DataFrame(replan_rows), use_container_width=True)

st.subheader("Raw trace (last 50 events)")
st.dataframe(pd.DataFrame(entries[-50:]), use_container_width=True)
