# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'schema and validation-flow verification: ranges, sentence counter, indicator mentions, repair then fallback', Date: 2026-10-06
"""Offline verification for schemas, validators, and the repair/fallback flow."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.analysis import analysis
from src.llm import prompts
from src.llm.llm_client import LLMValidationError
from src.schemas import (
    HeadlineSentiment,
    SentimentAggregate,
    SummaryStats,
    TechnicalSignal,
    count_sentences,
    indicator_concepts_mentioned,
)


def _signal(justification: str, action: str = "HOLD") -> TechnicalSignal:
    return TechnicalSignal(
        action=action, justification=justification,
        key_drivers=["sma structure"], risk_notes="n/a", generated_by="llm",
    )


def test_headline_sentiment_confidence_bounds():
    valid = HeadlineSentiment(
        headline="h", sentiment="positive", confidence=0.8, brief_reason="r"
    )
    assert valid.confidence == 0.8
    with pytest.raises(ValidationError):
        HeadlineSentiment(headline="h", sentiment="positive", confidence=1.5, brief_reason="r")
    with pytest.raises(ValidationError):
        HeadlineSentiment(headline="h", sentiment="positive", confidence=-0.1, brief_reason="r")
    with pytest.raises(ValidationError):
        HeadlineSentiment(headline="h", sentiment="bullish", confidence=0.5, brief_reason="r")


def test_sentence_counter_handles_decimals_and_abbreviations():
    assert count_sentences("One two three. Four five. Six.") == 3
    # 1.5 and U.S. must not inflate the count:
    text = "Revenue grew 1.5 times. The U.S. market strengthened. Margins held."
    assert count_sentences(text) == 3
    assert count_sentences("Only one sentence here") == 1


def test_signal_sentence_count_validator():
    good = (
        "Price holds above both moving averages. MACD histogram is positive and RSI "
        "is in the bullish band, so momentum aligns with the trend. Sentiment is not "
        "contrary. Risk stays contained."
    )
    assert count_sentences(good) == 4
    assert _signal(good).action == "HOLD"
    with pytest.raises(ValidationError):
        _signal("Only two sentences. Nothing more here.")

# -- combination-reasoning validator -------------------------------------------------

def test_signal_combination_reasoning_validator():
    four_sentences = (
        "First plain sentence. Second plain sentence. Third plain sentence. "
        "Fourth plain sentence."
    )
    assert indicator_concepts_mentioned(four_sentences) == 0
    with pytest.raises(ValidationError):
        _signal(four_sentences)  # no indicator concepts -> rejected


class FakeLLM:
    """Duck-typed client: scripted chat_json results/exceptions per call."""

    def __init__(self, script):
        import types

        self.script = list(script)
        self.model = "fake-model"
        self.settings = types.SimpleNamespace(model="fake-model")

    def chat_json(self, messages, model_cls, max_tokens=None):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return model_cls.model_validate(item)


def _summary(momentum: str = "bullish") -> SummaryStats:
    return SummaryStats.model_validate(
        {
            "ticker": "NVDA",
            "current_price": 100.0,
            "momentum_signal": momentum,
            "momentum_components": {
                "price_above_sma50": True, "price_above_sma200": True,
                "sma50_above_sma200": True, "macd_hist_positive": True,
                "rsi_in_bullish_band": True, "bullish_points": 5,
                "bearish_points": 0, "net": 5, "rsi_note": "",
            },
        }
    )


def _aggregate() -> SentimentAggregate:
    item = HeadlineSentiment(
        headline="h", sentiment="positive", confidence=0.9, brief_reason="r"
    )
    return SentimentAggregate(
        overall_score=0.5, label="positive",
        counts={"positive": 1, "negative": 0, "neutral": 0},
        top_headlines=[item], items=[item], fallback_count=0,
        model_id="m", generated_at="now",
    )


def test_generate_signal_uses_llm_when_valid():
    justification = (
        "Price holds above both moving averages while the MACD histogram stays "
        "positive, so the trend is confirmed by momentum. RSI inside the bullish "
        "band supports continuation rather than exhaustion. News sentiment is "
        "mildly positive, which does not contradict the technical picture. Risks "
        "remain contained near the 52-week high."
    )
    client = FakeLLM([{
        "action": "BUY", "justification": justification,
        "key_drivers": ["trend"], "risk_notes": "valuation", "generated_by": "llm",
    }])
    signal = analysis.generate_signal(_summary(), None, client)
    assert signal.action == "BUY" and signal.generated_by == "llm"


def test_generate_signal_falls_back_after_validation_failure():
    client = FakeLLM([LLMValidationError("schema mismatch"), LLMValidationError("again")])
    signal = analysis.generate_signal(_summary(), None, client)
    assert signal.generated_by == "rules_fallback"
    assert signal.action == "BUY"  # bullish momentum vote maps to BUY
    assert 3 <= count_sentences(signal.justification) <= 5
    assert indicator_concepts_mentioned(signal.justification) >= 2


def test_score_headlines_one_call_per_headline_and_sentinel():
    headlines = [{"title": "Good news"}, {"title": "Bad news"}, {"title": "Meh news"}]
    client = FakeLLM([
        {"headline": "Good news", "sentiment": "positive", "confidence": 0.9, "brief_reason": "r"},
        {"headline": "Bad news", "sentiment": "negative", "confidence": 0.3, "brief_reason": "r"},
        LLMValidationError("garbage"),
    ])
    aggregate = analysis.score_headlines("NVDA", headlines, client)
    assert len(aggregate.items) == 3
    assert aggregate.fallback_count == 1
    assert aggregate.items[2].fallback is True
    assert aggregate.items[2].sentiment == "neutral"
    assert aggregate.items[2].confidence == 0.0
    # overall = (0.9 - 0.3) / (0.9 + 0.3 + 0.0)
    assert aggregate.overall_score == pytest.approx(0.5)
    assert aggregate.counts == {"positive": 1, "negative": 1, "neutral": 1}
    assert aggregate.top_headlines[0].confidence == pytest.approx(0.9)


def test_score_headlines_zero_confidence_guard():
    client = FakeLLM([LLMValidationError("x")])
    aggregate = analysis.score_headlines("NVDA", [{"title": "h"}], client)
    assert aggregate.overall_score == 0.0  # sum(confidence) == 0 -> 0.0, no crash


def test_prompt_templates_format_against_real_payloads():
    assert prompts.SENTIMENT_USER.format(ticker="NVDA", headline="Some headline")
    bundle = analysis.build_evidence_bundle(_summary(), _aggregate())
    assert prompts.SIGNAL_USER.format(
        ticker="NVDA", as_of=bundle["as_of"], evidence_json=json.dumps(bundle)
    )
    assert prompts.REPAIR_SUFFIX.format(error="e", schema="{}")
