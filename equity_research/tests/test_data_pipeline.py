"""Offline verification for the data pipeline (mocked yfinance, no network)."""
from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.data import data_pipeline

def _synthetic_frame(periods: int = 800, seed: int = 11) -> pd.DataFrame:
    """Deterministic OHLCV frame ending today (runtime dates, never literals)."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    rng = np.random.default_rng(seed)
    close = np.clip(100.0 + np.cumsum(rng.normal(0.05, 1.0, periods)), 20.0, None)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
        },
        index=index,
    )

def _quality(frame: pd.DataFrame) -> dict:
    return {
        "ticker": "NVDA", "period": "3y", "bars": int(len(frame)),
        "missing_bars_dropped": 0,
        "first_date": frame.index[0].date().isoformat(),
        "last_date": frame.index[-1].date().isoformat(),
        "span_days": int((frame.index[-1] - frame.index[0]).days),
    }

def test_build_summary_populates_every_block():
    frame = _synthetic_frame()
    stats = data_pipeline.build_summary("nvda", frame, _quality(frame), {"trailingPE": 61.25})
    assert stats.ticker == "NVDA"
    assert stats.current_price == pytest.approx(float(frame["Close"].iloc[-1]))
    assert stats.prev_close == pytest.approx(float(frame["Close"].iloc[-2]))
    # the summary rounds percentages to 4 decimals for clean output
    assert stats.daily_return_pct == pytest.approx(
        (float(frame["Close"].iloc[-1]) / float(frame["Close"].iloc[-2]) - 1) * 100, abs=1e-3
    )
    assert stats.pe_trailing == pytest.approx(61.25)
    assert stats.indicators.sma_50 is not None and stats.indicators.sma_200 is not None
    assert stats.momentum_signal in {"bullish", "bearish", "neutral"}
    for key in ("bullish_points", "bearish_points", "net", "rsi_note"):
        assert key in stats.momentum_components
    assert stats.data_quality["bars"] == len(frame)

def test_nan_injection_still_yields_valid_summary():
    frame = _synthetic_frame()
    rng = np.random.default_rng(2)
    drop_positions = rng.choice(len(frame), size=40, replace=False)
    damaged = frame.copy()
    damaged.iloc[drop_positions, damaged.columns.get_loc("Close")] = np.nan
    damaged.iloc[drop_positions[:20], damaged.columns.get_loc("High")] = np.nan
    cleaned, quality = data_pipeline._clean(damaged, "NVDA", "3y")
    assert quality["missing_bars_dropped"] == 40
    assert quality["span_days"] >= data_pipeline.config.MIN_SPAN_DAYS
    stats = data_pipeline.build_summary("NVDA", cleaned, quality, {})
    assert stats.current_price is not None
    assert stats.indicators.sma_50 is not None

def test_ytd_uses_previous_year_close():
    frame = _synthetic_frame()
    stats = data_pipeline.build_summary("NVDA", frame, _quality(frame), {})
    last_year = frame.index[-1].year
    prior_mask = frame.index.year == (last_year - 1)
    assert prior_mask.any()
    prior_close = float(frame.loc[prior_mask, "Close"].iloc[-1])
    expected = (float(frame["Close"].iloc[-1]) / prior_close - 1) * 100
    assert stats.ytd_return_pct == pytest.approx(expected, abs=1e-3)  # 4-decimal rounding
    assert stats.ytd_note == ""

def test_pe_missing_renders_as_none():
    frame = _synthetic_frame()
    stats = data_pipeline.build_summary("NVDA", frame, _quality(frame), {})
    assert stats.pe_trailing is None  # report layer renders "N/A (source unavailable)"

def test_short_history_raises_market_data_error():
    frame = _synthetic_frame(periods=100)
    with pytest.raises(data_pipeline.MarketDataError):
        data_pipeline._clean(frame, "NVDA", "3y")

def _patched_fetch(monkeypatch, outcomes):
    calls = []

    def fake_fetch(ticker, period):
        calls.append(period)
        result = outcomes.pop(0) if outcomes else None
        if isinstance(result, Exception):
            raise result
        return result if result is not None else pd.DataFrame()

    monkeypatch.setattr(data_pipeline, "_fetch_once", fake_fetch)
    return calls

def test_fetch_ladder_falls_back_to_shorter_period(monkeypatch):
    good = _synthetic_frame()
    calls = _patched_fetch(monkeypatch, [None, None, None, good])  # 3 misses, fallback hit
    monkeypatch.setattr(data_pipeline.time, "sleep", lambda *_: None)
    frame, quality = data_pipeline.fetch_history("NVDA")
    assert quality["period"] == data_pipeline.config.FALLBACK_PERIOD
    assert calls[-1] == data_pipeline.config.FALLBACK_PERIOD
    assert len(frame) == len(good)

def test_fetch_ladder_exhaustion_raises(monkeypatch):
    _patched_fetch(monkeypatch, [None] * 8)
    monkeypatch.setattr(data_pipeline.time, "sleep", lambda *_: None)
    with pytest.raises(data_pipeline.MarketDataError):
        data_pipeline.fetch_history("NVDA")

def test_fetch_info_failure_degrades_to_empty(monkeypatch):
    class ExplodingTicker:
        def __init__(self, *_):
            raise RuntimeError("network down")

    monkeypatch.setitem(sys.modules, "yfinance", ModuleType("yfinance"))
    sys.modules["yfinance"].Ticker = ExplodingTicker  # type: ignore[attr-defined]
    assert data_pipeline.fetch_info("NVDA") == {}

def test_cached_summary_loader_flags_stale(tmp_path, monkeypatch):
    frame = _synthetic_frame()
    stats = data_pipeline.build_summary("NVDA", frame, _quality(frame), {})
    target = tmp_path / "summary.json"
    target.write_text(stats.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(data_pipeline.config, "SUMMARY_JSON", target)
    loaded = data_pipeline.load_cached_summary()
    assert loaded is not None and loaded.stale is True
    assert loaded.ticker == "NVDA"
