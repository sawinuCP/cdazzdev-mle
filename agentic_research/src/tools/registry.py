"""Registry and dispatcher: the single place tools get routed and digested."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from pydantic import ValidationError

from .. import config
from ..schemas import ToolResult
from . import base, news, pricing, sentiment, volatility, websearch
from ..runtime.llm_client import LLMClient

_TOOL_RUNNERS: Dict[str, Callable[..., ToolResult]] = {
    config.TOOL_GET_PRICE_DATA: pricing.get_price_data,
    config.TOOL_GET_NEWS: news.get_news,
    config.TOOL_CALCULATE_VOLATILITY: volatility.calculate_volatility,
    config.TOOL_LLM_SENTIMENT: sentiment.llm_sentiment,
    config.TOOL_WEB_SEARCH: websearch.web_search,
}

_ARG_MODELS: Dict[str, type] = {
    config.TOOL_GET_PRICE_DATA: pricing.GetPriceDataArgs,
    config.TOOL_GET_NEWS: news.GetNewsArgs,
    config.TOOL_CALCULATE_VOLATILITY: volatility.CalculateVolatilityArgs,
    config.TOOL_LLM_SENTIMENT: sentiment.LlmSentimentArgs,
    config.TOOL_WEB_SEARCH: websearch.WebSearchArgs,
}

_DIGESTERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    config.TOOL_GET_PRICE_DATA: pricing.price_digest,
    config.TOOL_GET_NEWS: news.news_digest,
    config.TOOL_CALCULATE_VOLATILITY: volatility.volatility_digest,
    config.TOOL_LLM_SENTIMENT: sentiment.sentiment_digest,
    config.TOOL_WEB_SEARCH: websearch.search_digest,
}

def tool_signature(name: str) -> str:
    """One-line signature used in the agent prompts."""
    signatures = {
        config.TOOL_GET_PRICE_DATA: "get_price_data(ticker: str, period: str)",
        config.TOOL_GET_NEWS: "get_news(ticker: str, n: int)",
        config.TOOL_CALCULATE_VOLATILITY: "calculate_volatility(ticker: str, window: int)",
        config.TOOL_LLM_SENTIMENT: "llm_sentiment(headlines: list[str])",
        config.TOOL_WEB_SEARCH: "web_search(query: str)",
    }
    return signatures.get(name, name)

def allowed_tools_block(whitelist) -> str:
    """Render the allowed tools (with signatures) for an agent prompt."""
    return "\n".join(f"- {tool_signature(name)}" for name in whitelist)

def dispatch(name: str, args: Dict[str, Any], llm: Optional[LLMClient],
             ticker: str = "") -> ToolResult:
    """Validate args, run the tool, and guarantee a ToolResult comes back."""
    if name not in _TOOL_RUNNERS:
        return base.make_result(False, error=f"unknown tool {name!r}",
                                hint=f"known tools: {', '.join(sorted(_TOOL_RUNNERS))}",
                                source="dispatcher")
    model = _ARG_MODELS[name]
    try:
        validated = model.model_validate(args)
    except ValidationError as exc:
        return base.make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                                hint=str(exc.errors()[0].get("input", ""))[:120],
                                source="dispatcher")
    runner = _TOOL_RUNNERS[name]
    if name == config.TOOL_LLM_SENTIMENT:
        if llm is None:
            return base.make_result(False, error="LLM client unavailable",
                                    hint="configure LLM_* environment variables", source="llm")
        return runner(validated.headlines, llm, ticker=ticker)
    if name == config.TOOL_GET_PRICE_DATA:
        return runner(validated.ticker, validated.period)
    if name == config.TOOL_GET_NEWS:
        return runner(validated.ticker, validated.n)
    if name == config.TOOL_CALCULATE_VOLATILITY:
        return runner(validated.ticker, validated.window)
    if name == config.TOOL_WEB_SEARCH:
        return runner(validated.query)
    return base.make_result(False, error="unroutable tool", source="dispatcher")

def digest_for(name: str, result: ToolResult) -> str:
    """One-line observation digest; prefixed with the failure when not ok."""
    if not result.ok:
        return f"FAILED: {result.error or 'unknown error'} (hint: {result.hint})"
    digester = _DIGESTERS.get(name)
    if digester is None:
        return json.dumps(result.data, ensure_ascii=False, default=str)
    try:
        return digester(result.data or {})
    except Exception:  # noqa: BLE001 - a digest must never break the loop
        return json.dumps(result.data, ensure_ascii=False, default=str)
