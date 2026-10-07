# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'tool verification: ToolResult contract, volatility math, news shapes and ladder, sentiment aggregation, web backoff', Date: 2026-10-06
"""Offline verification for the five tools (all data sources mocked)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config, tools
from src.runtime.llm_client import LLMSettings
from helpers import FakeLLM, synthetic_frame


@pytest.fixture
def fake_frame():
    return synthetic_frame()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(tools.websearch.time, "sleep", lambda *_: None)


# ── get_price_data ────────────────────────────────────────────────────────────
def test_get_price_data_compact_payload(fake_frame, monkeypatch):
    monkeypatch.setattr(tools.sources, "_fetch_history", lambda t, p: fake_frame)
    result = tools.get_price_data("NVDA", "1y")
    assert result.ok and result.source == "yfinance"
    data = result.data
    assert len(data["rows"]) == 5
    assert data["indicator_state"]["sma50_stance"] in {"above", "below"}
    assert data["week52_high"] == pytest.approx(float(fake_frame.tail(252)["High"].max()))
    assert result.fetched_at  # iso timestamp present


def test_get_price_data_rejects_bad_period():
    result = tools.get_price_data("NVDA", "7 PARSECS")
    assert not result.ok and "period" in (result.error or "")


def test_get_price_data_empty_frame_is_not_an_exception(monkeypatch):
    monkeypatch.setattr(tools.sources, "_fetch_history", lambda t, p: pd.DataFrame())
    result = tools.get_price_data("NVDA", "1y")
    assert not result.ok
    assert "shorter period" in (result.hint or "")


# ── calculate_volatility ──────────────────────────────────────────────────────
def test_volatility_exact_on_synthetic_series(fake_frame, monkeypatch):
    monkeypatch.setattr(tools.sources, "_fetch_history", lambda t, p: fake_frame)
    close = fake_frame["Close"].astype(float)
    window = 90
    log_returns = np.log(close / close.shift(1)).dropna().tail(window)
    expected_daily = float(log_returns.std(ddof=0))
    expected_annualized = expected_daily * np.sqrt(252)

    result = tools.calculate_volatility("NVDA", window)
    assert result.ok
    assert result.data["daily"] == pytest.approx(expected_daily, rel=1e-4)  # tool rounds to 6dp
    assert result.data["annualized"] == pytest.approx(expected_annualized, rel=1e-4)  # 4dp
    assert result.data["n_obs"] == window
    assert result.data["band"] in {"low", "moderate", "high"}


def test_volatility_window_validation():
    result = tools.calculate_volatility("NVDA", 1000)
    assert not result.ok and "window" in (result.hint or "")
    result = tools.calculate_volatility("NVDA", 1)
    assert not result.ok and "window" in (result.hint or "")


def test_volatility_band_thresholds(fake_frame, monkeypatch):
    monkeypatch.setattr(tools.sources, "_fetch_history", lambda t, p: fake_frame)
    result = tools.calculate_volatility("NVDA", 252)
    ann = result.data["annualized"]
    expected_band = ("low" if ann < config.VOL_BAND_LOW_MAX
                     else "high" if ann >= config.VOL_BAND_HIGH_MIN else "moderate")
    assert result.data["band"] == expected_band


# ── get_news ──────────────────────────────────────────────────────────────────
def test_news_normalizes_both_yfinance_shapes():
    import calendar
    import datetime

    epoch = calendar.timegm(datetime.datetime(2026, 10, 5, 12, 0).timetuple())
    flat = tools._normalize_yf_item({
        "title": "Legacy shape headline", "publisher": "Reuters",
        "link": "http://x/1", "providerPublishTime": epoch})
    nested = tools._normalize_yf_item({"content": {
        "title": "Nested shape headline", "provider": {"displayName": "Bloomberg"},
        "canonicalUrl": {"url": "http://x/2"}, "pubDate": "2026-10-05T09:30:00Z"}})
    assert flat["source"] == "Reuters" and flat["published"] == "2026-10-05"
    assert nested["source"] == "Bloomberg" and nested["url"] == "http://x/2"


def _rss(*titles: str) -> str:
    items = "".join(
        f"<item><title>{t}</title><link>http://x</link></item>" for t in titles
    )
    return (f'<?xml version="1.0"?><rss version="2.0"><channel>{items}</channel></rss>')


def test_news_ladder_falls_through_to_rss(monkeypatch):
    monkeypatch.setattr(tools.sources, "_news_raw", lambda t: [])
    monkeypatch.setattr(tools.sources, "_http_get",
                        lambda url: _rss("RSS headline A", "RSS headline B"))
    result = tools.get_news("NVDA", 2)
    assert result.ok and len(result.data["headlines"]) == 2
    assert result.data["sources"] == ["yahoo_rss"]


def test_news_zero_results_returns_failure_with_hint(monkeypatch):
    empty_feed = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
    monkeypatch.setattr(tools.sources, "_news_raw", lambda t: [])
    monkeypatch.setattr(tools.sources, "_http_get", lambda url: empty_feed)
    result = tools.get_news("NVDA", 10)
    assert not result.ok and "web_search" in (result.hint or "")


def test_news_coverage_labels(monkeypatch):
    monkeypatch.setattr(tools.sources, "_news_raw", lambda t: [])
    monkeypatch.setattr(tools.sources, "_http_get",
                        lambda url: _rss("H1", "H2", "H3"))
    assert tools.get_news("NVDA", 10).data["coverage"] == "low"
    assert tools.get_news("NVDA", 3).data["coverage"] == "full"

# ── llm_sentiment ─────────────────────────────────────────────────────────────
def _sentiment_llm(items):
    """FakeLLM answering per-headline sentiment calls in order."""
    payloads = [{"headline": h, "sentiment": s, "confidence": c, "brief_reason": "r"}
                for h, s, c in items]
    return FakeLLM(payloads)


def test_sentiment_one_call_per_headline_and_aggregation():
    llm = _sentiment_llm([("Good news", "positive", 0.9),
                          ("Bad news", "negative", 0.3),
                          ("Meh news", "neutral", 0.5)])
    result = tools.llm_sentiment(["Good news", "Bad news", "Meh news"], llm, ticker="NVDA")
    assert result.ok
    assert len(llm.calls) == 3  # ONE call per headline
    data = result.data
    # overall = (0.9 - 0.3) / (0.9 + 0.3 + 0.5) = 0.353 > +0.15 -> positive
    assert data["overall_score"] == pytest.approx(0.6 / 1.7, abs=1e-3)  # 4-decimal rounding
    assert data["label"] == "positive"
    assert data["counts"] == {"positive": 1, "negative": 1, "neutral": 1}


def test_sentiment_all_neutral_pulls_to_zero():
    llm = _sentiment_llm([("A", "neutral", 0.8), ("B", "neutral", 0.6)])
    data = tools.llm_sentiment(["A", "B"], llm).data
    assert data["overall_score"] == 0.0 and data["label"] == "neutral"


def test_sentiment_zero_confidence_guard():
    llm = FakeLLM([RuntimeError("gateway down"), RuntimeError("gateway down")])
    data = tools.llm_sentiment(["Only headline"], llm).data
    assert data["overall_score"] == 0.0
    assert data["estimates"] == 1  # neutral sentinel with confidence 0
    assert data["items"][0]["estimate"] is True


# ── web_search ────────────────────────────────────────────────────────────────
def test_web_search_returns_ranked_results(no_sleep, monkeypatch):
    monkeypatch.setattr(tools.sources, "_ddgs_text",
                        lambda q, max_results: [
                            {"title": "Analysts weigh in", "body": "snippet body",
                             "href": "http://x/1"}])
    result = tools.web_search("NVDA analyst commentary")
    assert result.ok
    assert result.data["results"][0]["url"] == "http://x/1"


def test_web_search_backoff_then_success(no_sleep, monkeypatch):
    calls = {"n": 0}

    def flaky(query, max_results):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return [{"title": "Recovered", "body": "b", "href": "http://x/2"}]

    monkeypatch.setattr(tools.sources, "_ddgs_text", flaky)
    result = tools.web_search("NVDA risks")
    assert result.ok and calls["n"] == 2  # one backoff retry happened


def test_web_search_failure_never_raises(no_sleep, monkeypatch):
    def dead(query, max_results):
        raise RuntimeError("down")

    monkeypatch.setattr(tools.sources, "_ddgs_text", dead)
    result = tools.web_search("NVDA risks")
    assert not result.ok and result.error


def test_web_search_bad_query_is_handled():
    result = tools.web_search("ab")  # below the minimum length
    assert not result.ok and "invalid arguments" in (result.error or "")
