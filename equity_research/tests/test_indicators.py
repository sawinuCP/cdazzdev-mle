# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'offline indicator verification: golden values, edges, properties, slow-loop cross-check', Date: 2026-10-06
"""Offline verification for the first-principles indicators (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis import indicators


def _series(values) -> pd.Series:
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=index)


# ── golden values ────────────────────────────────────────────────────────────
def test_sma_golden_values():
    out = indicators.sma(_series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])  # min_periods: no partial windows
    assert out.iloc[2] == pytest.approx(2.0)   # mean(1,2,3)
    assert out.iloc[3] == pytest.approx(3.0)   # mean(2,3,4)
    assert out.iloc[4] == pytest.approx(4.0)   # mean(3,4,5)


def test_rsi_hand_computed_wilder_value():
    # 13 up-moves of +1 then one down-move of -1: avg_gain=13/14, avg_loss=1/14,
    # RS=13 -> RSI = 100 - 100/14. Hand-derivable from the Wilder recursion.
    closes = [float(i) for i in range(14)] + [12.0]
    out = indicators.rsi(_series(closes), 14)
    assert np.isnan(out.iloc[13])                     # warm-up: only 13 deltas seen
    assert out.iloc[14] == pytest.approx(100.0 - 100.0 / 14.0)


def test_rsi_strictly_rising_reaches_100():
    out = indicators.rsi(_series([float(i) for i in range(30)]), 14)
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_series_is_50_without_crash():
    out = indicators.rsi(_series([5.0] * 30), 14)
    assert out.dropna().iloc[-1] == pytest.approx(50.0)

# ── independent slow-loop re-implementation (protects the vectorised code) ──
def _slow_wilder_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Explicit recursion mirroring pandas ewm(alpha=1/p, adjust=False) with
    NaN-skipping and min_periods=period (first valid output after ``period``
    non-NaN deltas)."""
    alpha = 1.0 / period
    deltas = np.diff(closes, prepend=np.nan)
    avg_gain = avg_loss = None
    observed = 0
    out = np.full(len(closes), np.nan)
    for i, delta in enumerate(deltas):
        if np.isnan(delta):
            continue
        up, down = max(delta, 0.0), max(-delta, 0.0)
        observed += 1
        if observed == 1:
            avg_gain, avg_loss = up, down
        else:
            avg_gain = (1 - alpha) * avg_gain + alpha * up
            avg_loss = (1 - alpha) * avg_loss + alpha * down
        if observed >= period:
            if avg_loss == 0 and avg_gain > 0:
                out[i] = 100.0
            elif avg_loss == 0 and avg_gain == 0:
                out[i] = 50.0
            else:
                out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _slow_sma(closes: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(window - 1, len(closes)):
        out[i] = closes[i - window + 1: i + 1].mean()
    return out


def test_vectorised_matches_slow_loops():
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0.0, 1.0, 400))
    series = _series(closes)

    vector_rsi = indicators.rsi(series).to_numpy()
    slow_rsi = _slow_wilder_rsi(closes)
    valid = np.isfinite(slow_rsi)
    np.testing.assert_allclose(vector_rsi[valid], slow_rsi[valid], atol=1e-8)

    for window in (config.SMA_SHORT_WINDOW, 20):
        vector_sma = indicators.sma(series, window).to_numpy()
        slow = _slow_sma(closes, window)
        np.testing.assert_allclose(vector_sma[np.isfinite(slow)], slow[np.isfinite(slow)], atol=1e-8)


# ── properties ───────────────────────────────────────────────────────────────
def test_rsi_stays_in_unit_range():
    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0.0, 1.5, 400))
    values = indicators.rsi(_series(closes)).dropna()
    assert ((values >= 0.0) & (values <= 100.0)).all()


def test_bollinger_geometry_and_constant_series():
    rng = np.random.default_rng(5)
    closes = 100 + np.cumsum(rng.normal(0.0, 1.0, 200))
    bands = indicators.bollinger(_series(closes))
    valid = bands.dropna()
    assert (valid["upper"] >= valid["mid"]).all() and (valid["mid"] >= valid["lower"]).all()

    flat = indicators.bollinger(_series([5.0] * 40))
    last = flat.iloc[-1]
    assert last["upper"] == pytest.approx(last["mid"])   # zero-width band, no crash
    assert last["lower"] == pytest.approx(last["mid"])
    assert np.isnan(last["pct_b"])          # position within a zero-width band: undefined
    assert last["bandwidth"] == pytest.approx(0.0)      # (upper-lower)/mid = 0/5 = 0


def test_macd_relationships_hold():
    rng = np.random.default_rng(9)
    series = _series(100 + np.cumsum(rng.normal(0.0, 1.0, 300)))
    frame = indicators.macd(series)
    ema_fast = series.ewm(span=config.MACD_FAST_SPAN, adjust=False).mean()
    ema_slow = series.ewm(span=config.MACD_SLOW_SPAN, adjust=False).mean()
    pd.testing.assert_series_equal(frame["macd"], ema_fast - ema_slow, check_names=False)
    pd.testing.assert_series_equal(
        frame["signal"], (ema_fast - ema_slow).ewm(span=config.MACD_SIGNAL_SPAN, adjust=False).mean(),
        check_names=False,
    )
    np.testing.assert_allclose(frame["hist"], frame["macd"] - frame["signal"], atol=1e-12)
    # MACD crosses zero exactly where EMA12 crosses EMA26
    pd.testing.assert_series_equal(
        np.sign(frame["macd"]), np.sign(ema_fast - ema_slow), check_names=False
    )


def test_macd_fresh_cross_flags_recent_sign_change():
    rising = [float(i) for i in range(40)]                      # histogram settles positive
    falling = [40.0 - 2.5 * i for i in range(1, 9)]             # steep reversal at the end
    frame = indicators.macd(_series(rising + falling))
    assert bool(frame["fresh_cross"].tail(8).any())


def test_compute_all_requires_history():
    with pytest.raises(indicators.InsufficientHistoryError):
        indicators.compute_all(_series([5.0] * 150))
    frame = indicators.compute_all(_series([float(i) for i in range(260)]))
    assert {"close", "sma_50", "sma_200", "rsi_14", "macd", "signal", "hist",
            "fresh_cross", "mid", "upper", "lower", "pct_b", "bandwidth"} <= set(frame.columns)
