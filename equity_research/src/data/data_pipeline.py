# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'market data pipeline with fetch ladder, cleaning, summary dictionary', Date: 2026-10-06
"""Market data pipeline: fetch, clean, and build the summary dictionary.

Failure philosophy: every problem becomes either a ``MarketDataError`` (which
the CLI handles with a friendly message and a stale-cache fallback) or a
logged, tolerated gap (P/E unavailable, a few null bars) — never a raw
traceback for the user.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from .. import config
from ..analysis import indicators

_LOGGER = logging.getLogger(__name__)

_OHLC_COLUMNS = ("Open", "High", "Low", "Close")


class MarketDataError(RuntimeError):
    """Raised when price history cannot be fetched or fails the sanity checks."""


def _fetch_once(ticker: str, period: str) -> pd.DataFrame:
    """One yfinance call. auto_adjust=False is deliberate: unadjusted closes
    match the textbook formulas the indicators implement (documented decision,
    not an accident)."""
    import yfinance as yf  # lazy import keeps unit tests light

    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def fetch_history(
    ticker: str, period: str = config.LOOKBACK_PERIOD
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch daily OHLCV through the failure ladder.

    Ladder: primary period with exponential-backoff retries -> shorter fallback
    period -> ``MarketDataError``. LOOKBACK_PERIOD is "3y" because the 200-day
    SMA consumes ~200 bars of warm-up; 3y keeps >= 2 years of data AND of valid
    indicator values, and the >= 730-day span is enforced here.
    """
    attempts = ((period, config.FETCH_RETRIES), (config.FALLBACK_PERIOD, 1))
    last_problem: str = "unknown"
    for attempt_period, retries in attempts:
        for attempt in range(retries):
            try:
                frame = _fetch_once(ticker, attempt_period)
            except Exception as exc:  # noqa: BLE001 - the ladder handles everything
                last_problem = f"{type(exc).__name__}: {exc}"
                _LOGGER.warning(
                    "fetch %s (%s) attempt %d failed: %s", ticker, attempt_period, attempt + 1, exc
                )
                time.sleep(config.FETCH_BACKOFF_BASE_S * (2 ** attempt))
                continue
            if frame.empty:
                last_problem = f"empty frame for {ticker} ({attempt_period})"
                _LOGGER.warning(last_problem)
                time.sleep(config.FETCH_BACKOFF_BASE_S * (2 ** attempt))
                continue
            return _clean(frame, ticker, attempt_period)
        _LOGGER.warning("falling back to a shorter fetch window for %s", ticker)
    raise MarketDataError(
        f"could not fetch price history for {ticker} (last problem: {last_problem})"
    )


def _clean(frame: pd.DataFrame, ticker: str, period: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enforce the history floor, drop null OHLC rows, and record data quality."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    span_days = int((frame.index[-1] - frame.index[0]).days)
    if span_days < config.MIN_SPAN_DAYS:
        raise MarketDataError(
            f"history for {ticker} spans only {span_days} days "
            f"({config.MIN_SPAN_DAYS} required)"
        )
    missing_bars = int(frame[list(_OHLC_COLUMNS)].isna().any(axis=1).sum())
    if missing_bars:
        frame = frame.dropna(subset=list(_OHLC_COLUMNS))
        _LOGGER.warning("dropped %d bars with null OHLC values for %s", missing_bars, ticker)
    quality = {
        "ticker": ticker,
        "period": period,
        "bars": int(len(frame)),
        "missing_bars_dropped": missing_bars,
        "first_date": frame.index[0].date().isoformat(),
        "last_date": frame.index[-1].date().isoformat(),
        "span_days": span_days,
    }
    _LOGGER.info(
        "fetched %s: %d bars %s..%s (span %d days)",
        ticker, quality["bars"], quality["first_date"], quality["last_date"], span_days,
    )
    return frame, quality


def fetch_info(ticker: str) -> Dict[str, Any]:
    """Ticker reference data (P/E etc.); failures degrade to an empty dict."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception as exc:  # noqa: BLE001 - reference data is optional
        _LOGGER.warning("ticker info unavailable for %s: %s", ticker, exc)
        return {}


def load_cached_summary() -> Optional["SummaryStats"]:
    """Load the previous summary (flagged stale) for the offline fallback path."""
    from ..schemas import SummaryStats  # local import avoids a circular dependency

    try:
        data = json_loads(config.SUMMARY_JSON.read_text(encoding="utf-8"))
        stats = SummaryStats.model_validate(data)
        stats.stale = True
        return stats
    except (OSError, ValidationError, ValueError):
        return None


def json_loads(text: str) -> Dict[str, Any]:
    """Tiny indirection so tests/monkeypatching stay simple."""
    import json

    return json.loads(text)

def build_summary(
    ticker: str,
    frame: pd.DataFrame,
    quality: Dict[str, Any],
    info: Optional[Dict[str, Any]] = None,
) -> "SummaryStats":
    """Assemble the clean summary dictionary and validate it into ``SummaryStats``."""
    from ..schemas import IndicatorSnapshot, SummaryStats  # local import: avoids cycles

    close = frame["Close"]
    ind = indicators.compute_all(close)
    last = ind.iloc[-1]

    price = _finite(close.iloc[-1])
    prev_close = _finite(close.iloc[-2]) if len(close) >= 2 else None
    daily_return_pct = (
        (price / prev_close - 1.0) * 100.0
        if price is not None and prev_close not in (None, 0.0)
        else None
    )

    week52 = frame.tail(config.WEEK52_BARS)
    week52_high = _finite(week52["High"].max())
    week52_low = _finite(week52["Low"].min())
    pct_from_52w_high = (
        (price / week52_high - 1.0) * 100.0
        if price is not None and week52_high not in (None, 0.0)
        else None
    )

    # Trailing P/E is optional reference data: any failure renders as "N/A" later.
    pe_trailing: Optional[float] = None
    try:
        raw_pe = (info or {}).get("trailingPE")
        pe_trailing = _finite(float(raw_pe)) if raw_pe is not None else None
    except (TypeError, ValueError):
        pe_trailing = None

    ytd_return_pct, ytd_note = _ytd_return(frame, price)

    snapshot = IndicatorSnapshot(
        price=price,
        sma_50=_finite(last.get("sma_50")),
        sma_200=_finite(last.get("sma_200")),
        price_vs_sma50=_stance(price, _finite(last.get("sma_50"))),
        price_vs_sma200=_stance(price, _finite(last.get("sma_200"))),
        sma50_vs_sma200=_stance(_finite(last.get("sma_50")), _finite(last.get("sma_200"))),
        rsi_14=_finite(last.get("rsi_14")),
        macd=_finite(last.get("macd")),
        macd_signal=_finite(last.get("signal")),
        macd_hist=_finite(last.get("hist")),
        macd_fresh_cross=bool(last.get("fresh_cross", False)),
        macd_hist_recent=[
            round(v, 4)
            for v in ind["hist"].tail(config.MACD_HIST_TREND_WINDOW).tolist()
            if pd.notna(v)
        ],
        pct_b=_finite(last.get("pct_b")),
        bandwidth=_finite(last.get("bandwidth")),
        rsi_note=_rsi_note(_finite(last.get("rsi_14"))),
    )

    momentum_signal, momentum_components = _momentum_vote(price, snapshot)

    return SummaryStats(
        ticker=ticker.upper(),
        current_price=price,
        prev_close=prev_close,
        daily_return_pct=_round(daily_return_pct),
        week52_high=week52_high,
        week52_low=week52_low,
        pct_from_52w_high=_round(pct_from_52w_high),
        pe_trailing=pe_trailing,
        ytd_return_pct=_round(ytd_return_pct),
        ytd_note=ytd_note,
        indicators=snapshot,
        momentum_signal=momentum_signal,
        momentum_components=momentum_components,
        data_quality=quality,
        stale=False,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _finite(value: Any) -> Optional[float]:
    """NaN/inf -> None (indicator warm-up must not leak into the summary)."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(as_float) else as_float


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


def _stance(value: Optional[float], reference: Optional[float]) -> str:
    if value is None or reference is None:
        return "unknown"
    return "above" if value > reference else "below"


def _rsi_note(rsi_value: Optional[float]) -> str:
    if rsi_value is None:
        return ""
    if rsi_value > config.RSI_OVERBOUGHT:
        return "overbought"
    if rsi_value < config.RSI_OVERSOLD:
        return "oversold"
    return ""

def _ytd_return(frame: pd.DataFrame, price: Optional[float]) -> Tuple[Optional[float], str]:
    """Year-to-date return: last close of the PREVIOUS calendar year -> latest close.

    Both endpoints are derived from the DatetimeIndex at runtime (no date
    literals). If the history lacks the prior-year close (e.g. a recent
    listing), fall back to the first close of the current year and note it —
    the previous-year definition is preferred because first-close-of-year
    understates YTD by one day's move.
    """
    if price is None or frame.empty:
        return None, ""
    last_date = frame.index[-1]
    prior_year_mask = frame.index.year == (last_date.year - 1)
    if prior_year_mask.any():
        prior_close = _finite(frame.loc[prior_year_mask, "Close"].iloc[-1])
        note = ""
    else:
        current_year_mask = frame.index.year == last_date.year
        prior_close = (
            _finite(frame.loc[current_year_mask, "Close"].iloc[0])
            if current_year_mask.any()
            else None
        )
        note = "prior-year close unavailable; used the first close of the current year"
    if prior_close in (None, 0.0):
        return None, note
    return (price / prior_close - 1.0) * 100.0, note


def _momentum_vote(
    price: Optional[float], snapshot: "IndicatorSnapshot"
) -> Tuple[str, Dict[str, Any]]:
    """Count bullish/bearish conditions (1 point each; RSI extremes are notes, not votes).

    Net >= +2 -> bullish, net <= -2 -> bearish, otherwise neutral. The full
    component dictionary is returned because the LLM reasons over it as
    evidence rather than receiving a bare verdict.
    """
    sma50, sma200 = snapshot.sma_50, snapshot.sma_200
    hist, rsi_value = snapshot.macd_hist, snapshot.rsi_14

    def _lt(value: Optional[float], threshold: float) -> bool:
        return value is not None and value < threshold

    def _gt(value: Optional[float], threshold: float) -> bool:
        return value is not None and value > threshold

    bullish_conditions = {
        "price_above_sma50": price is not None and sma50 is not None and price > sma50,
        "price_above_sma200": price is not None and sma200 is not None and price > sma200,
        "sma50_above_sma200": sma50 is not None and sma200 is not None and sma50 > sma200,
        "macd_hist_positive": hist is not None and hist > 0,
        "rsi_in_bullish_band": (
            rsi_value is not None
            and config.RSI_BULLISH_BAND[0] < rsi_value < config.RSI_BULLISH_BAND[1]
        ),
    }
    bearish_conditions = {
        "price_below_sma50": _lt(price, sma50) if sma50 is not None else False,
        "price_below_sma200": _lt(price, sma200) if sma200 is not None else False,
        "sma50_below_sma200": sma50 is not None and sma200 is not None and sma50 < sma200,
        "macd_hist_negative": _lt(hist, 0.0),
        "rsi_in_bearish_band": (
            rsi_value is not None
            and config.RSI_BEARISH_BAND[0] < rsi_value < config.RSI_BEARISH_BAND[1]
        ),
    }
    bullish_points = sum(1 for flag in bullish_conditions.values() if flag)
    bearish_points = sum(1 for flag in bearish_conditions.values() if flag)
    net = bullish_points - bearish_points

    components: Dict[str, Any] = dict(bullish_conditions)
    components.update(
        {
            "bullish_points": bullish_points,
            "bearish_points": bearish_points,
            "net": net,
            "rsi_note": snapshot.rsi_note,
        }
    )
    if net >= config.MOMENTUM_BULLISH_NET_THRESHOLD:
        signal = "bullish"
    elif net <= config.MOMENTUM_BEARISH_NET_THRESHOLD:
        signal = "bearish"
    else:
        signal = "neutral"
    return signal, components
