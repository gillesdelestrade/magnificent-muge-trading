"""Returns around the news and correlation indicators (with p-values and sample sizes)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

MIN_OBSERVATIONS = 30          # below: "échantillon insuffisant" instead of a conclusion
STRONG_SENTIMENT = 0.5
# Raw returns: r0 = day J (close J-1 -> close J, simultaneous reaction);
# r1, r5, r20 = following returns (close J -> close J+n). ar* = minus the benchmark.
HORIZONS = {"0": "Jour J", "1": "J+1", "5": "J+5", "20": "J+20"}
FORWARD = (1, 5, 20)


def _close(df: pd.DataFrame) -> pd.Series:
    return df["adj_close"] if "adj_close" in df.columns else df["close"]


def _windows(close: pd.Series, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out[f"{prefix}0"] = close / close.shift(1) - 1.0
    for n in FORWARD:
        out[f"{prefix}{n}"] = close.shift(-n) / close - 1.0
    return out


def returns_table(prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None) -> pd.DataFrame:
    """Per ticker and trading day: r0, r1, r5, r20 and the abnormal versions ar0…ar20."""
    frames = []
    for ticker, df in prices.items():
        if df is None or df.empty:
            continue
        close = _close(df)
        table = _windows(close, "r")
        if benchmark is not None and not benchmark.empty:
            bench = _close(benchmark)
            aligned = bench.reindex(bench.index.union(close.index)).ffill().reindex(close.index)
            market = _windows(aligned, "m")
            for key in HORIZONS:
                table[f"ar{key}"] = table[f"r{key}"] - market[f"m{key}"]
        else:
            for key in HORIZONS:
                table[f"ar{key}"] = np.nan
        table.insert(0, "ticker", ticker)
        table.index.name = "day"
        frames.append(table.reset_index())
    if not frames:
        return pd.DataFrame(columns=["day", "ticker"] + [f"{p}{k}" for p in ("r", "ar") for k in HORIZONS])
    return pd.concat(frames, ignore_index=True)


def correlation_stats(x: pd.Series, y: pd.Series) -> dict:
    """Pearson and Spearman correlations with p-values, sample size and hit rate.

    Hit rate: share of cases where the sentiment and the return have the same sign
    (cases where one of them is exactly zero are ignored).
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)
    out = {"n": n, "sufficient": n >= MIN_OBSERVATIONS, "pearson": None, "pearson_p": None,
           "spearman": None, "spearman_p": None, "hit_rate": None}
    if not out["sufficient"]:
        return out
    signed = data[(data["x"] != 0) & (data["y"] != 0)]
    if len(signed):
        out["hit_rate"] = float((np.sign(signed["x"]) == np.sign(signed["y"])).mean())
    if data["x"].nunique() > 1 and data["y"].nunique() > 1:
        pearson = stats.pearsonr(data["x"], data["y"])
        spearman = stats.spearmanr(data["x"], data["y"])
        out.update(pearson=float(pearson.statistic), pearson_p=float(pearson.pvalue),
                   spearman=float(spearman.statistic), spearman_p=float(spearman.pvalue))
    return out


def horizon_table(data: pd.DataFrame, sentiment_col: str = "sentiment",
                  abnormal: bool = True) -> pd.DataFrame:
    """Correlation indicators for each horizon (rows) of one data set."""
    prefix = "ar" if abnormal else "r"
    rows = [{"horizon": label, **correlation_stats(data[sentiment_col], data[f"{prefix}{key}"])}
            for key, label in HORIZONS.items()]
    return pd.DataFrame(rows).set_index("horizon")


def group_table(data: pd.DataFrame, group_col: str, return_col: str = "ar1",
                sentiment_col: str = "sentiment") -> pd.DataFrame:
    """Correlation indicators at one horizon for each value of ``group_col``."""
    rows = [{group_col: value, **correlation_stats(group[sentiment_col], group[return_col])}
            for value, group in data.groupby(group_col)]
    return pd.DataFrame(rows).set_index(group_col) if rows else pd.DataFrame()


def strong_days(data: pd.DataFrame, return_col: str = "ar1",
                threshold: float = STRONG_SENTIMENT) -> dict:
    """Mean following return on very positive vs very negative days."""
    valid = data.dropna(subset=["sentiment", return_col])
    positive = valid.loc[valid["sentiment"] >= threshold, return_col]
    negative = valid.loc[valid["sentiment"] <= -threshold, return_col]
    return {
        "n_positive": len(positive), "n_negative": len(negative),
        "mean_positive": float(positive.mean()) if len(positive) >= MIN_OBSERVATIONS else None,
        "mean_negative": float(negative.mean()) if len(negative) >= MIN_OBSERVATIONS else None,
    }


def split_by_cutoff(data: pd.DataFrame, cutoff: date | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(days up to the model's knowledge cutoff, days after). Unknown cutoff: everything
    is treated as possibly known by the model."""
    if cutoff is None:
        return data, data.iloc[0:0]
    limit = pd.Timestamp(cutoff)
    return data[data["day"] <= limit], data[data["day"] > limit]


def category_heatmap(articles: pd.DataFrame, abnormal: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation by news category x horizon, at article level.

    Returns (correlations, sample sizes); cells below MIN_OBSERVATIONS are NaN.
    """
    prefix = "ar" if abnormal else "r"
    corr, counts = {}, {}
    for category, group in articles.groupby("categorie"):
        corr[category], counts[category] = {}, {}
        for key, label in HORIZONS.items():
            s = correlation_stats(group["sentiment"], group[f"{prefix}{key}"])
            corr[category][label] = s["pearson"] if s["pearson"] is not None else np.nan
            counts[category][label] = s["n"]
    return (pd.DataFrame(corr).T.reindex(columns=list(HORIZONS.values())).astype("float64"),
            pd.DataFrame(counts).T.reindex(columns=list(HORIZONS.values())))
