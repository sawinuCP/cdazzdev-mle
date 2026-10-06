# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'analysis layer: per-headline sentiment with one call each and signal generation with fallback', Date: 2026-10-06
"""LLM analysis layer: per-headline sentiment and the recommendation signal.

Robustness contract: an LLM item that cannot be validated becomes a neutral
sentinel (confidence 0, ``fallback=True``, logged) instead of crashing; the
recommendation degrades to a deterministic rules-based signal derived from the
momentum vote. The LLM reasons over the evidence bundle — it never calculates.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config, prompts
from .llm_client import LLMClient, LLMError, LLMValidationError, failure_records
from .schemas import (
    HeadlineSentiment,
    SentimentAggregate,
    SummaryStats,
    TechnicalSignal,
)

_LOGGER = logging.getLogger(__name__)

_SENTINEL = "neutral"


def _sentiment_messages(ticker: str, headline: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompts.SENTIMENT_SYSTEM},
        {
            "role": "user",
            "content": prompts.SENTIMENT_USER.format(ticker=ticker.upper(), headline=headline),
        },
    ]


def _score_one(
    client: LLMClient, ticker: str, headline: str
) -> HeadlineSentiment:
    """One LLM call per headline (the specification is literal about this)."""
    try:
        value = client.chat_json(_sentiment_messages(ticker, headline), HeadlineSentiment)
        # Trust the caller's headline text over the model's echo of it.
        return value.model_copy(update={"headline": headline, "fallback": False})
    except LLMError as exc:
        _LOGGER.warning("headline sentiment failed (%s...): %s", headline[:60], exc)
        return HeadlineSentiment(
            headline=headline,
            sentiment=_SENTINEL,
            confidence=0.0,
            brief_reason="sentiment unavailable; neutral sentinel applied",
            fallback=True,
        )


def _aggregate_label(overall: float) -> str:
    if overall > config.SENTIMENT_POSITIVE_THRESHOLD:
        return "positive"
    if overall < config.SENTIMENT_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_headlines(ticker: str, headlines: List[Dict[str, Any]], client: LLMClient) -> SentimentAggregate:
    """Validate one headline per call, then aggregate with confidence weighting.

    Aggregate: ``overall = sum(sign * confidence) / sum(confidence)`` — neutral
    items carry sign 0 but stay in the denominator, so a neutral-heavy feed
    pulls the aggregate toward 0. If every confidence sums to zero, the overall
    value is 0.0.
    """
    items: List[HeadlineSentiment] = [
        _score_one(client, ticker, str(entry.get("title", "")).strip() or "(untitled headline)")
        for entry in headlines
    ]
    sign_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    weighted = sum(sign_map[item.sentiment] * item.confidence for item in items)
    total_confidence = sum(item.confidence for item in items)
    overall = weighted / total_confidence if total_confidence > 0 else 0.0
    overall = max(-1.0, min(1.0, overall))

    counts = {
        label: sum(1 for item in items if item.sentiment == label)
        for label in ("positive", "negative", "neutral")
    }
    top = sorted(items, key=lambda item: item.confidence, reverse=True)[
        : config.TOP_HEADLINES_COUNT
    ]
    aggregate = SentimentAggregate(
        overall_score=round(overall, 4),
        label=_aggregate_label(overall),
        counts=counts,
        top_headlines=top,
        items=items,
        fallback_count=sum(1 for item in items if item.fallback),
        model_id=client.settings.model,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _LOGGER.info(
        "sentiment: overall=%.3f (%s), counts=%s, fallbacks=%d",
        aggregate.overall_score, aggregate.label, counts, aggregate.fallback_count,
    )
    return aggregate

def build_evidence_bundle(summary: SummaryStats, sentiment: SentimentAggregate) -> Dict[str, Any]:
    """Assemble the evidence bundle the signal prompt reasons over.

    The 5-day MACD-histogram trend is labelled here ("rising" / "falling" /
    "turning" / "insufficient") from the recent histogram values stored in the
    summary — the LLM interprets trends, it does not compute them.
    """
    recent = summary.indicators.macd_hist_recent
    trend = "insufficient"
    if len(recent) >= 2:
        change = recent[-1] - recent[0]
        signs = {1 if v > 0 else -1 if v < 0 else 0 for v in recent}
        if len(signs) > 1:
            trend = "turning"
        elif change > 0:
            trend = "rising"
        elif change < 0:
            trend = "falling"
        else:
            trend = "flat"
    return {
        "ticker": summary.ticker,
        "as_of": summary.data_quality.get("last_date", ""),
        "price": summary.current_price,
        "prev_close": summary.prev_close,
        "daily_return_pct": summary.daily_return_pct,
        "week52_high": summary.week52_high,
        "week52_low": summary.week52_low,
        "pct_from_52w_high": summary.pct_from_52w_high,
        "ytd_return_pct": summary.ytd_return_pct,
        "indicators": summary.indicators.model_dump(),
        "momentum_signal": summary.momentum_signal,
        "momentum_components": summary.momentum_components,
        "macd_hist_trend_5d": trend,
        "news_sentiment": {
            "overall_score": sentiment.overall_score,
            "label": sentiment.label,
            "counts": sentiment.counts,
            "top_headlines": [
                {
                    "headline": item.headline,
                    "sentiment": item.sentiment,
                    "confidence": item.confidence,
                }
                for item in sentiment.top_headlines
            ],
        },
    }


def _signal_messages(ticker: str, bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    import json as _json

    return [
        {"role": "system", "content": prompts.SIGNAL_SYSTEM},
        {
            "role": "user",
            "content": prompts.SIGNAL_USER.format(
                ticker=ticker.upper(),
                as_of=bundle.get("as_of", ""),
                evidence_json=_json.dumps(bundle, indent=2, ensure_ascii=False),
            ),
        },
    ]


def _rules_fallback(summary: SummaryStats, sentiment: Optional[SentimentAggregate]) -> TechnicalSignal:
    """Deterministic recommendation derived from the momentum vote.

    The justification is crafted to satisfy the same validators as the LLM path
    (3-5 sentences, at least two indicator concepts related to each other), so a
    fallback never silently degrades the output contract.
    """
    action_map = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}
    action = action_map[summary.momentum_signal]
    components = summary.momentum_components
    drivers = [name for name, flag in components.items() if flag is True]
    sentiment_text = (
        f"News sentiment reads {sentiment.label} at {sentiment.overall_score:+.2f}, "
        f"which {'supports' if sentiment.label == ('positive' if action == 'BUY' else 'negative' if action == 'SELL' else 'neutral') else 'cuts against'} the technical vote."
        if sentiment
        else "News sentiment was unavailable, so this recommendation rests on price structure alone."
    )
    justification = (
        f"Price is {summary.indicators.price_vs_sma50} the 50-day SMA and "
        f"{summary.indicators.price_vs_sma200} the 200-day SMA, with the 50-day "
        f"{summary.indicators.sma50_vs_sma200} the 200-day, so the moving-average structure "
        f"{'confirms the trend' if summary.momentum_signal != 'neutral' else 'is mixed'}. "
        f"The MACD histogram is {'positive' if components.get('macd_hist_positive') else 'negative'} "
        f"and RSI sits in the {summary.indicators.rsi_note or 'mid'} zone, so momentum "
        f"{'aligns with' if summary.momentum_signal == 'bullish' else 'does not contradict' if summary.momentum_signal == 'neutral' else 'confirms pressure against'} the trend. "
        f"{sentiment_text} "
        "Size any position with respect to the distance from the 52-week high."
    )
    return TechnicalSignal(
        action=action,
        justification=justification,
        key_drivers=drivers[: config.TOP_HEADLINES_COUNT] or [summary.momentum_signal],
        risk_notes=(
            "Rules-based fallback generated without LLM reasoning; treat as a "
            "mechanical summary of the momentum vote, not a full analysis."
        ),
        generated_by="rules_fallback",
    )


def generate_signal(
    summary: SummaryStats,
    sentiment: Optional[SentimentAggregate],
    client: LLMClient,
) -> TechnicalSignal:
    """Recommend BUY/HOLD/SELL; degrade to the rules fallback if validation fails."""
    bundle = build_evidence_bundle(summary, sentiment) if sentiment else {
        "ticker": summary.ticker,
        "as_of": summary.data_quality.get("last_date", ""),
        "indicators": summary.indicators.model_dump(),
        "momentum_signal": summary.momentum_signal,
        "momentum_components": summary.momentum_components,
        "news_sentiment": {"note": "unavailable"},
    }
    try:
        signal = client.chat_json(_signal_messages(summary.ticker, bundle), TechnicalSignal)
        _LOGGER.info("signal: %s (llm)", signal.action)
        return signal
    except LLMValidationError as exc:
        _LOGGER.warning("signal validation failed after repair; using rules fallback: %s", exc)
        return _rules_fallback(summary, sentiment)
    except LLMError as exc:
        _LOGGER.warning("signal call failed; using rules fallback: %s", exc)
        return _rules_fallback(summary, sentiment)
