"""Pydantic models: every LLM response passes through one of these before use.

The validators enforce the required output properties mechanically — sentence
counts, indicator-mention checks, confidence ranges — so compliance does not
depend on the model choosing to obey.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

def _finite_or_none(value: Any) -> Optional[float]:
    """Convert NaN/inf (indicator warm-up) to None so JSON stays clean."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(as_float) or math.isinf(as_float) else as_float

def count_sentences(text: str) -> int:
    """Count sentences using a capital-letter look-ahead split.

    The regex splits only where a terminator is followed by whitespace and a
    capital, so decimals ("1.5") and abbreviations ("U.S.") do not inflate the
    count.
    """
    parts = re.split(config.SENTENCE_SPLIT_REGEX, text.strip())
    return len([p for p in parts if p.strip()])

def indicator_concepts_mentioned(text: str) -> int:
    """Number of distinct indicator concepts named in the text (case-insensitive)."""
    lowered = text.lower()
    return sum(1 for concept in config.INDICATOR_CONCEPTS if concept in lowered)

class IndicatorSnapshot(BaseModel):
    """Latest values of the computed indicators (NaN warm-up becomes None)."""

    price: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma50: Literal["above", "below", "unknown"] = "unknown"
    price_vs_sma200: Literal["above", "below", "unknown"] = "unknown"
    sma50_vs_sma200: Literal["above", "below", "unknown"] = "unknown"
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_fresh_cross: bool = False
    macd_hist_recent: List[float] = Field(default_factory=list)
    pct_b: Optional[float] = None
    bandwidth: Optional[float] = None
    rsi_note: Literal["", "overbought", "oversold"] = ""

    @field_validator(
        "price", "sma_50", "sma_200", "rsi_14", "macd", "macd_signal",
        "macd_hist", "pct_b", "bandwidth", mode="before",
    )
    @classmethod
    def _nan_to_none(cls, v: Any) -> Optional[float]:
        return _finite_or_none(v)

    @field_validator("macd_hist_recent", mode="before")
    @classmethod
    def _clean_recent(cls, v: Any) -> List[float]:
        if not isinstance(v, (list, tuple)):
            return []
        cleaned = [_finite_or_none(x) for x in v]
        return [x for x in cleaned if x is not None]

class SummaryStats(BaseModel):
    """The clean summary dictionary produced by the data pipeline."""

    ticker: str
    current_price: Optional[float] = None
    prev_close: Optional[float] = None
    daily_return_pct: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    pe_trailing: Optional[float] = None  # None renders as "N/A (source unavailable)"
    ytd_return_pct: Optional[float] = None
    ytd_note: str = ""
    indicators: IndicatorSnapshot = Field(default_factory=IndicatorSnapshot)
    momentum_signal: Literal["bullish", "bearish", "neutral"] = "neutral"
    momentum_components: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    stale: bool = False  # True when served from a cached file after a data failure
    generated_at: str = ""

    @field_validator(
        "current_price", "prev_close", "daily_return_pct", "week52_high",
        "week52_low", "pct_from_52w_high", "pe_trailing", "ytd_return_pct",
        mode="before",
    )
    @classmethod
    def _nan_to_none(cls, v: Any) -> Optional[float]:
        return _finite_or_none(v)

class HeadlineSentiment(BaseModel):
    """Structured sentiment for ONE headline (one LLM call per headline)."""

    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = ""
    fallback: bool = False  # True when the sentinel replaced a failed LLM item

class SentimentAggregate(BaseModel):
    """Per-headline items plus the confidence-weighted aggregate."""

    overall_score: float = Field(ge=-1.0, le=1.0)
    label: Literal["positive", "negative", "neutral"]
    counts: Dict[str, int] = Field(default_factory=dict)
    top_headlines: List[HeadlineSentiment] = Field(default_factory=list)
    items: List[HeadlineSentiment] = Field(default_factory=list)
    fallback_count: int = 0
    model_id: str = ""
    generated_at: str = ""

class TechnicalSignal(BaseModel):
    """Buy/Hold/Sell recommendation with a mechanically validated justification."""

    action: Literal["BUY", "HOLD", "SELL"]
    justification: str
    key_drivers: List[str] = Field(default_factory=list)
    risk_notes: str = ""
    generated_by: Literal["llm", "rules_fallback"]

    @field_validator("justification")
    @classmethod
    def _sentence_count(cls, v: str) -> str:
        n = count_sentences(v)
        if not (config.SENTENCE_COUNT_MIN <= n <= config.SENTENCE_COUNT_MAX):
            raise ValueError(
                f"justification must contain {config.SENTENCE_COUNT_MIN}-"
                f"{config.SENTENCE_COUNT_MAX} sentences, found {n}"
            )
        return v

    @model_validator(mode="after")
    def _combination_reasoning(self) -> "TechnicalSignal":
        mentioned = indicator_concepts_mentioned(self.justification)
        if mentioned < config.MIN_INDICATOR_MENTIONS:
            raise ValueError(
                "justification must relate at least "
                f"{config.MIN_INDICATOR_MENTIONS} indicator concepts to each other "
                f"(found {mentioned}); restating isolated values is not acceptable"
            )
        return self
