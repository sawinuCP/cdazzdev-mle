"""llm_sentiment(headlines): per-headline LLM classification + aggregation."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..runtime.llm_client import LLMClient
from ..prompts import SENTIMENT_SYSTEM, SENTIMENT_USER
from ..schemas import HeadlineSentiment, ToolResult
from .base import make_result

class LlmSentimentArgs(BaseModel):
    headlines: List[str] = Field(min_length=1)

def llm_sentiment(headlines: List[str], llm: LLMClient, ticker: str = "") -> ToolResult:
    """Score each headline separately, then aggregate with confidence weighting."""
    try:
        args = LlmSentimentArgs(headlines=headlines[:config.SENTIMENT_MAX_HEADLINES])
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint="headlines must be a non-empty list of strings",
                           source="llm")
    items: List[Dict[str, Any]] = []
    for headline in args.headlines:
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM},
            {"role": "user", "content": SENTIMENT_USER.format(
                ticker=ticker.upper(), headline=headline)},
        ]
        try:
            value = llm.complete_json(messages, HeadlineSentiment)
            value = value.model_copy(update={"headline": headline})
            items.append(value.model_dump())
        except Exception:  # noqa: BLE001 - one bad headline never kills the batch
            items.append({"headline": headline, "sentiment": "neutral", "confidence": 0.0,
                          "brief_reason": "sentiment unavailable; neutral sentinel",
                          "estimate": True})
    sign_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    weighted = sum(sign_map[item["sentiment"]] * item["confidence"] for item in items)
    total_confidence = sum(item["confidence"] for item in items)
    overall = weighted / total_confidence if total_confidence > 0 else 0.0
    overall = max(-1.0, min(1.0, overall))
    if overall > config.SENTIMENT_POSITIVE_THRESHOLD:
        label = "positive"
    elif overall < config.SENTIMENT_NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    counts = {name: sum(1 for item in items if item["sentiment"] == name)
              for name in ("positive", "negative", "neutral")}
    payload = {
        "items": items, "overall_score": round(overall, 4), "label": label,
        "counts": counts,
        "estimates": sum(1 for item in items if item.get("estimate")),
    }
    return make_result(True, data=payload, source="llm")

def sentiment_digest(data: Dict[str, Any]) -> str:
    return (f"sentiment={data.get('label')} ({data.get('overall_score'):+.3f}) "
            f"counts={data.get('counts')} estimates={data.get('estimates')}")
