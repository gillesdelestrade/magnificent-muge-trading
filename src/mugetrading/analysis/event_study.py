"""Event study: mean cumulative abnormal return around strong news, from J-5 to J+20."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation import STRONG_SENTIMENT, _close

BEFORE, AFTER = 5, 20


def abnormal_daily_returns(prices: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.Series:
    """Daily return of the stock minus the daily return of the benchmark."""
    close = _close(prices)
    stock = close.pct_change()
    if benchmark is None or benchmark.empty:
        return stock
    bench = _close(benchmark)
    aligned = bench.reindex(bench.index.union(close.index)).ffill().reindex(close.index)
    return stock - aligned.pct_change()


def event_study(daily: pd.DataFrame, prices: dict[str, pd.DataFrame],
                benchmark: pd.DataFrame | None, threshold: float = STRONG_SENTIMENT,
                before: int = BEFORE, after: int = AFTER) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mean cumulative abnormal return around strong positive and negative news.

    Returns (frame indexed by the offset J-before..J+after with columns ``positive`` and
    ``negative``, number of events of each kind). The cumulative return starts at 0 the
    day before the window. Events too close to the edges of the price history are skipped.
    """
    offsets = np.arange(-before, after + 1)
    paths: dict[str, list[np.ndarray]] = {"positive": [], "negative": []}
    for ticker, group in daily.groupby("ticker"):
        if ticker not in prices:
            continue
        abnormal = abnormal_daily_returns(prices[ticker], benchmark)
        positions = {day: i for i, day in enumerate(abnormal.index)}
        values = abnormal.to_numpy()
        for row in group.itertuples():
            kind = ("positive" if row.sentiment >= threshold
                    else "negative" if row.sentiment <= -threshold else None)
            i = positions.get(row.day)
            if kind is None or i is None or i - before < 1 or i + after >= len(values):
                continue
            window = values[i - before:i + after + 1]
            if np.isnan(window).any():
                continue
            paths[kind].append(np.cumsum(window))
    result = pd.DataFrame(index=pd.Index(offsets, name="offset"))
    for kind, items in paths.items():
        result[kind] = np.mean(items, axis=0) if items else np.nan
    return result, {kind: len(items) for kind, items in paths.items()}
