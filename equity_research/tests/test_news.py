# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline news verification: both yfinance shapes, RSS parsing, ladder fallback, dedupe, coverage', Date: 2026-10-06
"""Offline verification for the news ladder (mocked network)."""
from __future__ import annotations

import pytest

from src import news


def _headline(title: str, source: str = "Test") -> dict:
    return {"title": title, "source": source, "published": "", "url": "http://example.com"}


def test_normalizes_legacy_flat_shape():
    item = {
        "title": "Chipmaker raises guidance",
        "publisher": "Reuters",
        "link": "http://example.com/a",
        "providerPublishTime": 0,  # replaced below with a real epoch
    }
    import calendar, datetime
    epoch = calendar.timegm(datetime.datetime(2026, 10, 5, 12, 0).timetuple())
    item["providerPublishTime"] = epoch
    entry = news._normalize_yf_item(item)
    assert entry is not None
    assert entry["title"] == "Chipmaker raises guidance"
    assert entry["source"] == "Reuters"
    assert entry["url"] == "http://example.com/a"
    assert entry["published"] == "2026-10-05"


def test_normalizes_new_nested_shape():
    item = {
        "content": {
            "title": "Analysts weigh in on the chip cycle",
            "provider": {"displayName": "Bloomberg"},
            "canonicalUrl": {"url": "http://example.com/b"},
            "pubDate": "2026-10-05T09:30:00Z",
        }
    }
    entry = news._normalize_yf_item(item)
    assert entry == {
        "title": "Analysts weigh in on the chip cycle",
        "source": "Bloomberg",
        "published": "2026-10-05",
        "url": "http://example.com/b",
    }


def test_normalizer_skips_empty_titles():
    assert news._normalize_yf_item({"title": ""}) is None
    assert news._normalize_yf_item("not-a-dict") is None


def test_rss_parsing_and_pubdate_normalization():
    xml_text = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>Alpha beats &amp; raises</title><link>http://x/1</link>"
        "<pubDate>Mon, 05 Oct 2026 10:00:00 +0000</pubDate>"
        "<source>Yahoo Finance RSS</source></item>"
        "<item><title>Beta merger announced</title><link>http://x/2</link>"
        "<pubDate>Tue, 06 Oct 2026 09:00:00 +0000</pubDate></item>"
        "</channel></rss>"
    )
    items = news._parse_rss(xml_text, "Yahoo Finance RSS")
    assert [i["title"] for i in items] == ["Alpha beats & raises", "Beta merger announced"]
    assert items[0]["published"] == "2026-10-05"
    assert items[1]["source"] == "Yahoo Finance RSS"


def test_title_key_dedupes_and_unescapes():
    assert news._normalize_title_key("Alpha  Beats! &amp; Raises") == news._normalize_title_key(
        "alpha beats raises"
    )


def test_ladder_falls_through_to_rss(monkeypatch):
    monkeypatch.setattr(news, "_fetch_yfinance", lambda ticker: [])
    monkeypatch.setattr(
        news, "_fetch_yahoo_rss",
        lambda ticker: [_headline(f"Headline {i}", "Yahoo") for i in range(12)],
    )
    result = news.fetch_headlines("NVDA")
    assert len(result["headlines"]) == 12
    assert result["coverage"] == "full"
    assert result["sources"] == ["yahoo_rss"]


def test_ladder_survives_failing_first_source(monkeypatch):
    def boom(_ticker):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(news, "_fetch_yfinance", boom)
    monkeypatch.setattr(news, "_fetch_yahoo_rss", lambda ticker: [_headline("Only one")])
    monkeypatch.setattr(news, "_fetch_google_rss", lambda ticker: [_headline("Second")])
    result = news.fetch_headlines("NVDA")
    assert len(result["headlines"]) == 2
    assert result["coverage"] == "low"
    assert result["sources"] == ["yahoo_rss", "google_rss"]


def test_ladder_dedupes_across_sources(monkeypatch):
    monkeypatch.setattr(news, "_fetch_yfinance", lambda ticker: [_headline("Same story", "A")])
    monkeypatch.setattr(
        news, "_fetch_yahoo_rss",
        lambda ticker: [_headline("Same Story!", "B"), _headline("Different one", "B")],
    )
    monkeypatch.setattr(news, "_fetch_google_rss", lambda ticker: [])
    result = news.fetch_headlines("NVDA")
    assert [item["title"] for item in result["headlines"]] == ["Same story", "Different one"]


def test_ladder_caps_headlines(monkeypatch):
    monkeypatch.setattr(
        news, "_fetch_yfinance",
        lambda ticker: [_headline(f"Story number {i}") for i in range(30)],
    )
    result = news.fetch_headlines("NVDA")
    assert len(result["headlines"]) == news.config.MAX_HEADLINES


def test_ladder_returns_partial_label(monkeypatch):
    monkeypatch.setattr(news, "_fetch_yfinance", lambda ticker: [])
    monkeypatch.setattr(
        news, "_fetch_yahoo_rss",
        lambda ticker: [_headline(f"Item {i}") for i in range(news.config.PARTIAL_COVERAGE_MIN)],
    )
    monkeypatch.setattr(news, "_fetch_google_rss", lambda ticker: [])
    result = news.fetch_headlines("NVDA")
    assert result["coverage"] == "partial"
    assert 0 < len(result["headlines"]) < news.config.MIN_HEADLINES
