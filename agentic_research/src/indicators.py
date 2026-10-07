# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'self-contained copy of the indicator functions for the agent tool', Date: 2026-10-06
# SOURCE: equity_research/src/indicators.py (own code), Lines 18-149
"""Technical indicators computed from first principles — no TA-Lib.

Deliberately copied (not imported) from ``equity_research/src/indicators.py``
so this folder stays self-contained for reviewers. Conventions: Wilder RSI via
``ewm(alpha=1/n, adjust=False)``; population sigma (ddof=0) for Bollinger;
``min_periods`` prevents partial-window averages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class InsufficientHistoryError(ValueError):
    """Raised when the price history is too short for the requested indicators."""


def sma(close: pd.Series, window: int = config.SMA_SHORT_WINDOW) -> pd.Series:
    """Simple moving average over ``window`` closes (no partial windows)."""
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """RSI with Wilder's smoothing (alpha = 1/period).

    Guards: pure up-moves -> 100; a perfectly flat series -> 50.
    """
    alpha = 1.0 / period
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    values = 100.0 - 100.0 / (1.0 + rs)
    values = values.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    values = values.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return values


def macd(close: pd.Series, fast_span: int = config.MACD_FAST_SPAN,
         slow_span: int = config.MACD_SLOW_SPAN,
         signal_span: int = config.MACD_SIGNAL_SPAN) -> pd.DataFrame:
    """MACD line, signal line (EMA of the MACD series), histogram, fresh cross."""
    ema_fast = close.ewm(span=fast_span, adjust=False).mean()
    ema_slow = close.ewm(span=slow_span, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    hist = macd_line - signal_line
    sign = np.sign(hist)
    changed = pd.Series(sign.diff().fillna(0.0) != 0, index=close.index)
    fresh_cross = (
        changed.rolling(window=config.FRESH_CROSS_WINDOW, min_periods=1)
        .max().fillna(0.0).astype(bool)
    )
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist, "fresh_cross": fresh_cross},
        index=close.index,
    )


def bollinger(close: pd.Series, window: int = config.BB_WINDOW,
              num_std: float = config.BB_NUM_STD) -> pd.DataFrame:
    """Bollinger Bands (population sigma), %B and bandwidth (zero-width guarded)."""
    mid = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    band_range = upper - lower
    pct_b = (close - lower) / band_range.replace(0.0, np.nan)
    bandwidth = band_range / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth},
        index=close.index,
    )


def compute_all(close: pd.Series) -> pd.DataFrame:
    """Compute every indicator with a minimum-history guard."""
    required = config.MIN_BARS_FOR_INDICATORS
    if len(close) < required:
        raise InsufficientHistoryError(
            f"price history has {len(close)} bars; at least {required} are required"
        )
    frame = pd.DataFrame({"close": close})
    frame["sma_50"] = sma(close, config.SMA_SHORT_WINDOW)
    frame["sma_200"] = sma(close, config.SMA_LONG_WINDOW)
    frame["rsi_14"] = rsi(close, config.RSI_PERIOD)
    frame = frame.join(macd(close))
    frame = frame.join(bollinger(close))
    return frame
