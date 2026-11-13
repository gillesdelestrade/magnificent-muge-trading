"""Buy signals: momentum and breakout. Values at day J only use data up to day J."""
from __future__ import annotations

import pandas as pd

from .params import COMBINE_AND, COMBINE_BREAKOUT, COMBINE_MOMENTUM, StrategyParams

REASON_MOMENTUM = "Momentum"
REASON_BREAKOUT = "Breakout"
REASON_BOTH = "Momentum + Breakout"


def momentum(close: pd.Series, period: int) -> pd.Series:
    """close J / close J-period - 1"""
    return close / close.shift(period) - 1.0


def breakout_level(high: pd.Series, period: int) -> pd.Series:
    """Highest high of the ``period`` days before J (J excluded)."""
    return high.shift(1).rolling(period, min_periods=period).max()


def compute_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Signals for one stock. ``df`` has (adjusted) high and close columns.

    Returns columns: momentum, breakout_level, momentum_signal, breakout_signal,
    buy_signal, reason.
    """
    out = pd.DataFrame(index=df.index)
    out["momentum"] = momentum(df["close"], params.momentum_period)
    out["breakout_level"] = breakout_level(df["high"], params.breakout_period)
    out["momentum_signal"] = (out["momentum"] >= params.momentum_threshold).fillna(False)
    out["breakout_signal"] = (
        df["close"] > out["breakout_level"] * (1.0 + params.breakout_margin)
    ).fillna(False)

    mom, brk = out["momentum_signal"], out["breakout_signal"]
    if params.combination == COMBINE_MOMENTUM:
        buy = mom
    elif params.combination == COMBINE_BREAKOUT:
        buy = brk
    elif params.combination == COMBINE_AND:
        buy = mom & brk
    else:
        buy = mom | brk
    out["buy_signal"] = buy

    reason = pd.Series("", index=df.index, dtype="object")
    if params.uses_momentum:
        reason[mom] = REASON_MOMENTUM
    if params.uses_breakout:
        reason[brk] = REASON_BREAKOUT
    if params.uses_momentum and params.uses_breakout:
        reason[mom & brk] = REASON_BOTH
    out["reason"] = reason.where(buy, "")
    return out
