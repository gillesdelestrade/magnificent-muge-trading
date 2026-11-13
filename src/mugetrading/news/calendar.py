"""Attach each article to a trading day.

An article published after the close (17:30, Paris time) or on a non-trading day
(week-end, holiday) belongs to the NEXT trading day: it could not move the price before.
"""
from __future__ import annotations

from datetime import time

import pandas as pd

from .provider import PARIS

MARKET_CLOSE = time(17, 30)


def assign_trading_days(published_at: pd.Series, trading_days: pd.DatetimeIndex,
                        close: time = MARKET_CLOSE) -> pd.Series:
    """Trading day of each publication time (NaT when after the last known trading day).

    ``trading_days`` are the days on which the stock was actually quoted (tz-naive,
    sorted): holidays are therefore handled without any holiday calendar.
    """
    if len(published_at) == 0:
        return pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    local = pd.to_datetime(published_at, utc=True).dt.tz_convert(PARIS)
    day = local.dt.tz_localize(None).dt.normalize()
    after_close = local.dt.time > close
    candidate = day + pd.to_timedelta(after_close.astype(int), unit="D")
    days = pd.DatetimeIndex(trading_days).sort_values()
    position = days.searchsorted(candidate.to_numpy(), side="left")
    result = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    valid = position < len(days)
    result[valid] = days[position[valid]]
    return result
