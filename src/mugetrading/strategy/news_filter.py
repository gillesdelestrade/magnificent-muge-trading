"""News filter: buy / sell decisions drawn from the AI analysis of the articles.

Values at day J only use the articles attached to day J or before (an article published
after the close belongs to the next trading day, see news/calendar.py). The engine then
shifts them by one bar, like the price signals: decided at the close of J, executed at the
open of J+1. No AI call is made here: only analyses already stored are read.
"""
from __future__ import annotations

import pandas as pd

from .params import StrategyParams

REASON_NEGATIVE_NEWS = "Actualité négative"
NEWS_COLUMNS = ["ticker", "day", "sentiment", "pertinence", "confiance"]


def news_signals(index: pd.DatetimeIndex, articles: pd.DataFrame | None,
                 params: StrategyParams) -> pd.DataFrame:
    """Per trading day of one stock: news_sentiment (weighted mean over the last N trading
    days, NaN without any news), buy_ok and sell.

    ``articles``: analysed articles of this stock (columns day, sentiment, pertinence,
    confiance), one row per article.
    """
    out = pd.DataFrame({"news_sentiment": float("nan"), "buy_ok": True, "sell": False}, index=index)
    if not params.news_filter_enabled:
        return out
    if articles is None or articles.empty:
        out["buy_ok"] = params.news_buy_without_news
        return out

    weight = articles["pertinence"] * articles["confiance"]
    daily = pd.DataFrame({"day": articles["day"], "weight": weight,
                          "weighted": articles["sentiment"] * weight}).groupby("day").sum()
    daily = daily.reindex(index, fill_value=0.0)
    window = daily.rolling(params.news_buy_window, min_periods=1).sum()
    has_news = window["weight"] > 1e-12
    out.loc[has_news, "news_sentiment"] = window["weighted"][has_news] / window["weight"][has_news]

    if params.news_sell_enabled:
        strong = articles[(articles["sentiment"] <= params.news_sell_threshold)
                          & (articles["pertinence"] >= params.news_sell_min_pertinence)]
        out["sell"] = index.isin(strong["day"])
    out["buy_ok"] = ((out["news_sentiment"] >= params.news_buy_threshold).where(
        has_news, params.news_buy_without_news)) & ~out["sell"]
    return out
