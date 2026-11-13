"""Join articles with their AI analysis and aggregate the sentiment per stock and trading day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..news.calendar import assign_trading_days

ARTICLE_TABLE_COLUMNS = ["article_id", "ticker", "day", "published_at", "source", "title", "url",
                         "sentiment", "pertinence", "categorie", "surprise", "horizon",
                         "confiance", "resume_fr", "provider_sentiment"]


def article_table(articles: pd.DataFrame, results: pd.DataFrame,
                  prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per analysed (article, ticker), attached to its trading day.

    Articles published after the close or on a closed day go to the next trading day;
    those after the last known trading day are dropped.
    """
    if articles.empty or results.empty:
        return pd.DataFrame(columns=ARTICLE_TABLE_COLUMNS)
    merged = articles.rename(columns={"id": "article_id"}).merge(
        results, on=["article_id", "ticker"], how="inner")
    merged["day"] = pd.NaT
    merged["day"] = merged["day"].astype("datetime64[ns]")
    for ticker, group in merged.groupby("ticker"):
        if ticker in prices and not prices[ticker].empty:
            merged.loc[group.index, "day"] = assign_trading_days(
                group["published_at"], prices[ticker].index)
    merged = merged.dropna(subset=["day"])
    return merged[ARTICLE_TABLE_COLUMNS].sort_values(["day", "ticker"]).reset_index(drop=True)


def daily_sentiment(table: pd.DataFrame) -> pd.DataFrame:
    """Sentiment of the day = mean of the sentiments weighted by pertinence x confiance.

    Columns: ticker, day, sentiment, n_articles, provider_sentiment (plain mean of the
    score given by the news source, NaN if none). Days whose articles all have a zero
    weight (irrelevant) are dropped.
    """
    columns = ["ticker", "day", "sentiment", "n_articles", "provider_sentiment"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    work = table.assign(weight=table["pertinence"] * table["confiance"])
    work["weighted"] = work["sentiment"] * work["weight"]
    grouped = work.groupby(["ticker", "day"], as_index=False).agg(
        weighted=("weighted", "sum"), weight=("weight", "sum"),
        n_articles=("article_id", "nunique"), provider_sentiment=("provider_sentiment", "mean"))
    grouped = grouped[grouped["weight"] > 0]
    grouped["sentiment"] = grouped["weighted"] / grouped["weight"]
    grouped["provider_sentiment"] = grouped["provider_sentiment"].astype("float64").replace({np.nan: np.nan})
    return grouped[columns].reset_index(drop=True)
