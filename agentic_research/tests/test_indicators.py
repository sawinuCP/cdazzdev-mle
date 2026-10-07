# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'indicator verification: golden values, edges, properties, relationships', Date: 2026-10-06
"""Offline verification for the copied indicators (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.tools import indicators


def _series(values) -> pd.Series:
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=index)


def test_sma_golden_values():
    out = indicators.sma(_series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_hand_computed_wilder_value():
    closes = [float(i) for i in range(14)] + [12.0]
    out = indicators.rsi(_series(closes), 14)
    assert np.isnan(out.iloc[13])
    assert out.iloc[14] == pytest.approx(100.0 - 100.0 / 14.0)  # RS = 13


def test_rsi_edges():
    assert indicators.rsi(_series([float(i) for i in range(30)])).dropna().iloc[-1] \
        == pytest.approx(100.0)
    assert indicators.rsi(_series([5.0] * 30)).dropna().iloc[-1] == pytest.approx(50.0)


def test_rsi_bounds_on_random_walk():
    rng = np.random.default_rng(3)
    values = indicators.rsi(_series(100 + np.cumsum(rng.normal(0, 1.5, 400)))).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_bollinger_ordering_and_constant_series():
    rng = np.random.default_rng(5)
    bands = indicators.bollinger(_series(100 + np.cumsum(rng.normal(0, 1, 200))))
    valid = bands.dropna()
    assert (valid["upper"] >= valid["mid"]).all() and (valid["mid"] >= valid["lower"]).all()
    flat = indicators.bollinger(_series([5.0] * 40)).iloc[-1]
    assert flat["upper"] == pytest.approx(flat["mid"]) and np.isnan(flat["pct_b"])


def test_macd_relationships():
    rng = np.random.default_rng(9)
    series = _series(100 + np.cumsum(rng.normal(0, 1, 300)))
    frame = indicators.macd(series)
    np.testing.assert_allclose(frame["hist"], frame["macd"] - frame["signal"], atol=1e-12)
    ema_fast = series.ewm(span=config.MACD_FAST_SPAN, adjust=False).mean()
    ema_slow = series.ewm(span=config.MACD_SLOW_SPAN, adjust=False).mean()
    pd.testing.assert_series_equal(frame["macd"], ema_fast - ema_slow, check_names=False)
