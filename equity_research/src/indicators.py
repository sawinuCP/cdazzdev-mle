# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'first-principles SMA/RSI/MACD/Bollinger with guards and tests-friendly purity', Date: 2026-10-06
"""Technical indicators computed from first principles — no TA-Lib, no indicator library.

Why pandas ``rolling``/``ewm`` still counts as "first principles": the requirement bans
*indicator libraries*, not vectorised arithmetic. Every formula below is the textbook
definition expressed with basic pandas primitives, and the tests re-derive the same
numbers with an independent slow loop.

Key decisions (also stated at each function):
- RSI uses Wilder's smoothing, i.e. an exponential recursion with ``alpha = 1/n``.
  Using ``span=n`` instead would apply a different decay constant and silently
  shift RSI levels; ``alpha=1/n`` is the canonical Wilder definition.
- ``adjust=False`` seeds the recursion from the first observation; the difference
  from an SMA-seeded start vanishes after the warm-up window and pandas exposes
  the recursion directly.
- Bollinger uses the population standard deviation (``ddof=0``), the conventional
  charting choice, so band widths match what analysts expect to see.
All functions are pure: a Series goes in, a Series/DataFrame comes out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class InsufficientHistoryError(ValueError):
    """Raised when the price history is too short for the requested indicators."""


def sma(close: pd.Series, window: int = config.SMA_SHORT_WINDOW) -> pd.Series:
    """Simple moving average: mean of the last ``window`` closes.

    ``min_periods=window`` prevents partial-window averages, which would fake
    indicator values early in the history.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (alpha = 1/period).

    Guards:
    - ``avg_loss == 0`` and ``avg_gain > 0``  -> RSI = 100 (pure up-moves).
    - both averages zero (e.g. a perfectly flat series) -> RSI = 50, by
      convention, so the indicator degrades gracefully instead of NaN-ing out.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")
    alpha = 1.0 / period  # Wilder's decay; span=period would be a different filter
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss  # float division; 0/0 -> NaN, x/0 -> inf
    values = 100.0 - 100.0 / (1.0 + rs)
    values = values.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    values = values.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return values

def macd(
    close: pd.Series,
    fast_span: int = config.MACD_FAST_SPAN,
    slow_span: int = config.MACD_SLOW_SPAN,
    signal_span: int = config.MACD_SIGNAL_SPAN,
) -> pd.DataFrame:
    """MACD line, signal line, histogram and a "fresh cross" flag.

    - MACD   = EMA(fast) - EMA(slow), each via ``ewm(span=..., adjust=False)``.
    - Signal = EMA(signal_span) OF THE MACD SERIES (the published definition —
      smoothing the *MACD line*, not the closes).
    - Histogram = MACD - Signal.
    - ``fresh_cross`` is True when the histogram changed sign within the last
      ``config.FRESH_CROSS_WINDOW`` bars — a cheap, well-defined recency flag.
    """
    for name, span in (("fast", fast_span), ("slow", slow_span), ("signal", signal_span)):
        if span <= 0:
            raise ValueError(f"{name} span must be a positive integer")
    ema_fast = close.ewm(span=fast_span, adjust=False).mean()
    ema_slow = close.ewm(span=slow_span, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    hist = macd_line - signal_line

    sign = np.sign(hist)
    changed = pd.Series(sign.diff().fillna(0.0) != 0, index=close.index)
    fresh_cross = (
        changed.rolling(window=config.FRESH_CROSS_WINDOW, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(bool)
    )
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist, "fresh_cross": fresh_cross},
        index=close.index,
    )


def bollinger(
    close: pd.Series,
    window: int = config.BB_WINDOW,
    num_std: float = config.BB_NUM_STD,
) -> pd.DataFrame:
    """Bollinger Bands: SMA(window) +/- num_std * population sigma (ddof=0).

    Also returns ``pct_b`` — the close's position inside the band — and
    ``bandwidth``. Both guard the zero-width-band case (constant prices) by
    returning NaN instead of dividing by zero.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")
    mid = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)  # population sigma
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
    """Compute every indicator and return one aligned DataFrame.

    Raises ``InsufficientHistoryError`` when the history cannot support the
    longest-warm-up indicator (the 200-day SMA); callers convert this into a
    handled, user-friendly condition instead of a raw traceback.
    """
    required = config.SMA_LONG_WINDOW + 1
    if len(close) < required:
        raise InsufficientHistoryError(
            f"price history has {len(close)} bars; at least {required} are required "
            f"for the {config.SMA_LONG_WINDOW}-day SMA"
        )
    frame = pd.DataFrame({"close": close})
    frame["sma_50"] = sma(close, config.SMA_SHORT_WINDOW)
    frame["sma_200"] = sma(close, config.SMA_LONG_WINDOW)
    frame["rsi_14"] = rsi(close, config.RSI_PERIOD)
    frame = frame.join(macd(close))
    frame = frame.join(bollinger(close))
    return frame
