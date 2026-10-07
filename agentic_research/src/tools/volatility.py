"""calculate_volatility(ticker, window): annualized realized volatility."""
from __future__ import annotations

import logging
import math

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..schemas import ToolResult
from . import sources
from .base import make_result

_LOGGER = logging.getLogger(__name__)

class CalculateVolatilityArgs(BaseModel):
    ticker: str = Field(min_length=1)
    window: int = Field(default=config.VOL_WINDOW_DEFAULT, ge=config.VOL_WINDOW_MIN,
                        le=config.VOL_WINDOW_MAX)

def calculate_volatility(ticker: str, window: int) -> ToolResult:
    """Annualized realized volatility over the last ``window`` daily log returns."""
    try:
        args = CalculateVolatilityArgs(ticker=ticker, window=window)
    except ValidationError as exc:
        return make_result(False, error=f"invalid arguments: {exc.errors()[0]['msg']}",
                           hint=f"window must be between {config.VOL_WINDOW_MIN} "
                                f"and {config.VOL_WINDOW_MAX}",
                           source="yfinance")
    try:
        frame = sources._fetch_history(args.ticker, "2y")  # own fetch: never borrow another tool's state
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"{type(exc).__name__}: {exc}",
                           hint="try web_search for volatility commentary", source="yfinance")
    if frame.empty or "Close" not in frame:
        return make_result(False, error="empty price frame",
                           hint="try web_search for volatility commentary", source="yfinance")
    close = frame["Close"].astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < args.window:
        args.window = len(log_returns)  # degrade gracefully, report the real n_obs
    used = log_returns.tail(args.window)
    daily = float(used.std(ddof=0))
    annualized = daily * math.sqrt(config.TRADING_DAYS_PER_YEAR)
    if annualized < config.VOL_BAND_LOW_MAX:
        band = "low"
    elif annualized >= config.VOL_BAND_HIGH_MIN:
        band = "high"
    else:
        band = "moderate"
    payload = {
        "ticker": args.ticker.upper(), "annualized": round(annualized, 4),
        "daily": round(daily, 6), "window": args.window,
        "n_obs": int(len(used)), "band": band,
    }
    return make_result(True, data=payload, source="yfinance")

def volatility_digest(data: dict) -> str:
    return (f"vol annualized={data.get('annualized')} ({data.get('band')}) "
            f"daily={data.get('daily')} window={data.get('window')} n={data.get('n_obs')}")
