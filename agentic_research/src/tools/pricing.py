"""get_price_data(ticker, period): daily OHLCV tail + indicator snapshot."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from .. import config
from . import indicators
from ..schemas import ToolResult
from .base import _finite, make_result
from . import sources  # noqa: F401 _fetch_history

class GetPriceDataArgs(BaseModel):
    ticker: str = Field(min_length=1)
    period: str = config.DEFAULT_PERIOD

def get_price_data(ticker: str, period: str) -> ToolResult:
    """Daily OHLCV tail plus an indicator snapshot (built in Python)."""
    try:
        args = GetPriceDataArgs(ticker=ticker, period=period)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"period must be one of {', '.join(config.ALLOWED_PERIODS)}",
                           source="yfinance")
    if args.period not in config.ALLOWED_PERIODS:
        return make_result(False, error=f"period {args.period!r} not allowed",
                           hint=f"period must be one of {', '.join(config.ALLOWED_PERIODS)}",
                           source="yfinance")
    try:
        frame = sources._fetch_history(args.ticker, args.period)
    except Exception as exc:  # noqa: BLE001 - converted into a handled outcome
        return make_result(False, error=f"{type(exc).__name__}: {exc}",
                           hint="try a shorter period or web_search", source="yfinance")
    if frame.empty:
        return make_result(False, error="empty price frame",
                           hint="try a shorter period or web_search", source="yfinance")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().dropna(subset=["Open", "High", "Low", "Close"])
    close = frame["Close"].astype(float)
    price = _finite(close.iloc[-1])
    prev_close = _finite(close.iloc[-2]) if len(close) >= 2 else None

    indicator_state: Dict[str, Any] = {"sma50_stance": "unknown", "sma200_stance": "unknown",
                                       "rsi": None, "macd_hist": None,
                                       "macd_fresh_cross": False, "pct_b": None}
    try:
        ind = indicators.compute_all(close)
        last = ind.iloc[-1]
        sma50, sma200 = _finite(last.get("sma_50")), _finite(last.get("sma_200"))

        def _stance(value: Any, reference: Any) -> str:
            return "unknown" if value is None or reference is None else \
                ("above" if value > reference else "below")

        indicator_state = {
            "sma50_stance": _stance(price, sma50),
            "sma200_stance": _stance(price, sma200),
            "rsi": _finite(last.get("rsi_14")),
            "macd_hist": _finite(last.get("hist")),
            "macd_fresh_cross": bool(last.get("fresh_cross", False)),
            "pct_b": _finite(last.get("pct_b")),
        }
    except indicators.InsufficientHistoryError:
        import logging

        logging.getLogger(__name__).info(
            "price history too short for full indicators (%d bars)", len(close))

    week52 = frame.tail(config.WEEK52_BARS)
    week52_high, week52_low = _finite(week52["High"].max()), _finite(week52["Low"].min())
    ytd_pct = None
    last_date = frame.index[-1]
    prior_mask = frame.index.year == (last_date.year - 1)
    if prior_mask.any() and price is not None:
        prior_close = _finite(frame.loc[prior_mask, "Close"].iloc[-1])
        if prior_close not in (None, 0.0):
            ytd_pct = round((price / prior_close - 1.0) * 100.0, 2)

    rows = [
        {"date": idx.date().isoformat(),
         "open": round(_finite(row["Open"]) or 0.0, 2),
         "high": round(_finite(row["High"]) or 0.0, 2),
         "low": round(_finite(row["Low"]) or 0.0, 2),
         "close": round(_finite(row["Close"]) or 0.0, 2),
         "volume": int(row["Volume"])}
        for idx, row in frame.tail(5).iterrows()
    ]
    payload = {
        "ticker": args.ticker.upper(), "period": args.period,
        "current_price": price, "previous_close": prev_close,
        "week52_high": week52_high, "week52_low": week52_low, "ytd_pct": ytd_pct,
        "rows": rows, "indicator_state": indicator_state,
    }
    return make_result(True, data=payload, source="yfinance")

def price_digest(data: Dict[str, Any]) -> str:
    """Compact observation digest for the agent's context window."""
    state = data.get("indicator_state", {})
    return (f"price={data.get('current_price')} prev={data.get('previous_close')} "
            f"vs SMA50={state.get('sma50_stance')} vs SMA200={state.get('sma200_stance')} "
            f"RSI={state.get('rsi')} macd_hist={state.get('macd_hist')} "
            f"%B={state.get('pct_b')} 52w={data.get('week52_low')}-{data.get('week52_high')} "
            f"YTD={data.get('ytd_pct')}%")
