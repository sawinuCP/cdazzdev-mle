# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'pydantic schemas: ToolResult, AgentAction, brief, critique, research report with validators', Date: 2026-10-06
"""Pydantic models for the multi-agent research system.

Two honesty guarantees are enforced here mechanically:
- every numeric field in ``AgentBrief`` is built in Python from tool data
  (the LLM only writes ``analyst_notes``), and
- ``ResearchReport`` cannot validate without exactly three evidence-backed
  risks and a hedge grounded in at least two data points.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolResult(BaseModel):
    """Uniform tool outcome: tools NEVER raise, they return ok=False instead."""

    ok: bool
    data: Any = None
    error: Optional[str] = None
    hint: Optional[str] = None
    source: str = ""
    fetched_at: str = ""


class AgentAction(BaseModel):
    """One step of the JSON-action protocol returned by the LLM."""

    thought: str = ""
    action: str
    args: Dict[str, Any] = Field(default_factory=dict)
    replan_reason: Optional[str] = None


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = ""
    estimate: bool = False  # True when a sentinel replaced a failed LLM item


class SentimentBlock(BaseModel):
    """Aggregated per-headline sentiment (built in Python from tool data)."""

    overall_score: float
    label: str
    counts: Dict[str, int]
    items: List[HeadlineSentiment]


class PriceLevel(BaseModel):
    current: Optional[float] = None
    previous_close: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    ytd_pct: Optional[float] = None


class VolatilityState(BaseModel):
    annualized_90d: Optional[float] = None
    annualized_30d: Optional[float] = None
    daily: Optional[float] = None
    band: str = "unknown"


class IndicatorState(BaseModel):
    sma50_stance: str = "unknown"   # above | below | unknown
    sma200_stance: str = "unknown"
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_fresh_cross: bool = False
    pct_b: Optional[float] = None


class AgentBrief(BaseModel):
    """Typed handoff from the quantitative agent to the research writer.

    Every numeric field is computed in Python from ToolResult data; the LLM
    contributes ``analyst_notes`` only. The round trip must be lossless:
    ``AgentBrief.model_validate_json(brief.model_dump_json()) == brief``.
    """

    ticker: str
    as_of: str
    price_level: PriceLevel
    volatility: VolatilityState
    indicator_state: IndicatorState
    sentiment: Optional[SentimentBlock] = None  # None in the first brief
    analyst_notes: str = ""
    sources: List[str] = Field(default_factory=list)


class CritiqueQuestion(BaseModel):
    question_id: str
    text: str


class CritiqueRequest(BaseModel):
    """The research writer's structured request back to the data analyst."""

    request_id: str
    questions: List[CritiqueQuestion] = Field(min_length=1, max_length=2)
    payload: Dict[str, Any] = Field(default_factory=dict)  # headlines attached


class ClarificationAnswer(BaseModel):
    question_id: str
    response: str
    data: Dict[str, Any]


class ClarificationResponse(BaseModel):
    request_id: str
    answers: List[ClarificationAnswer] = Field(min_length=1)


class Evidence(BaseModel):
    source_tool: str
    key_datum: str
    url: Optional[str] = None


class Risk(BaseModel):
    title: str
    detail: str
    likelihood: int = Field(ge=1, le=5)
    evidence: List[Evidence] = Field(min_length=1)


class HedgePlan(BaseModel):
    strategy: str
    instruments: List[str] = Field(min_length=1)
    rationale: str
    data_basis: List[str] = Field(min_length=2)
    cost_note: str = ""


class ReportMeta(BaseModel):
    run_id: str
    run_at: str
    tool_call_count: int
    replans: int
    critique_cycles: int
    memory_hits: int
    degraded: bool = False


class ComposedReport(BaseModel):
    """What the LLM composes (no meta: run counters are Python-owned)."""

    ticker: str
    financial_health_summary: FinancialHealthSummary
    risks: List[Risk] = Field(min_length=3, max_length=3)
    hedge: HedgePlan
    meta_note: str = ""


class FinancialHealthSummary(BaseModel):
    narrative: str
    evidence_refs: List[str] = Field(min_length=1)


class ResearchReport(BaseModel):
    """The validated final artifact."""

    ticker: str
    financial_health_summary: FinancialHealthSummary
    risks: List[Risk] = Field(min_length=3, max_length=3)
    hedge: HedgePlan
    meta: ReportMeta
    disclaimer: str = (
        "This report is generated automatically for informational purposes only "
        "and is not investment advice."
    )

    @model_validator(mode="after")
    def _hedge_instruments_present(self) -> "ResearchReport":
        if not self.hedge.instruments:
            raise ValueError("hedge must name at least one instrument")
        return self
