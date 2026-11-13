"""Correlation and event-study maths checked on synthetic data with known results."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mugetrading.analysis.correlation import (MIN_OBSERVATIONS, category_heatmap,
                                              correlation_stats, horizon_table, returns_table,
                                              split_by_cutoff, strong_days)
from mugetrading.analysis.event_study import event_study
from mugetrading.analysis.sentiment_daily import article_table, daily_sentiment
from mugetrading.news.provider import normalize_articles


def frame(closes, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    return pd.DataFrame({"close": np.asarray(closes, dtype="float64")}, index=idx)


def test_returns_and_abnormal_returns():
    stock = frame([100, 110, 121, 121, 121, 121, 133.1])
    bench = frame([100, 105, 105, 105, 105, 105, 105])
    t = returns_table({"AAA": stock}, bench).set_index("day")
    d1 = stock.index[1]
    assert t.loc[d1, "r0"] == pytest.approx(0.10)          # close J-1 -> close J
    assert t.loc[d1, "r1"] == pytest.approx(0.10)          # close J -> close J+1
    assert t.loc[d1, "r5"] == pytest.approx(133.1 / 110 - 1)
    assert t.loc[d1, "ar0"] == pytest.approx(0.05)         # minus the index (+5 %)
    assert t.loc[d1, "ar1"] == pytest.approx(0.10)         # index flat the next day
    assert np.isnan(t.loc[stock.index[-1], "r1"])          # no future data available


def test_perfect_correlations():
    x = pd.Series(np.linspace(-1, 1, 40))
    s = correlation_stats(x, 0.02 * x)
    assert s["n"] == 40 and s["pearson"] == pytest.approx(1.0) and s["spearman"] == pytest.approx(1.0)
    assert s["pearson_p"] < 1e-6 and s["hit_rate"] == 1.0
    s = correlation_stats(x, -(x ** 3))
    assert s["spearman"] == pytest.approx(-1.0) and -1 < s["pearson"] < -0.8
    assert s["hit_rate"] == 0.0


def test_known_hit_rate_and_p_value():
    x = pd.Series([1.0, -1.0] * 20)
    y = pd.Series([0.01, -0.01] * 15 + [-0.01, 0.01] * 5)      # 30 agree, 10 disagree
    s = correlation_stats(x, y)
    assert s["hit_rate"] == pytest.approx(0.75) and s["pearson"] == pytest.approx(0.5)
    assert s["pearson_p"] == pytest.approx(0.001, abs=0.0005)   # r = 0.5 with n = 40


def test_insufficient_sample_gives_no_conclusion():
    x = pd.Series(np.linspace(-1, 1, MIN_OBSERVATIONS - 1))
    s = correlation_stats(x, x)
    assert s["n"] == 29 and not s["sufficient"]
    assert s["pearson"] is None and s["hit_rate"] is None
    days = pd.DataFrame({"sentiment": [0.9] * 5 + [-0.9] * 40, "ar1": [0.01] * 5 + [-0.02] * 40})
    strong = strong_days(days)
    assert strong["n_positive"] == 5 and strong["mean_positive"] is None
    assert strong["mean_negative"] == pytest.approx(-0.02)


def _articles_and_results():
    # Mon 08/01/2024 10:00 and 18:00 (after close), Sat 13/01 -> Mon 15/01
    articles = normalize_articles(pd.DataFrame([
        {"published_at": "2024-01-08T10:00:00+01:00", "source": "s", "title": "a", "text": "",
         "url": "u1", "ticker": "AAA", "provider_sentiment": 0.2},
        {"published_at": "2024-01-08T11:00:00+01:00", "source": "s", "title": "b", "text": "",
         "url": "u2", "ticker": "AAA", "provider_sentiment": 0.6},
        {"published_at": "2024-01-08T18:00:00+01:00", "source": "s", "title": "c", "text": "",
         "url": "u3", "ticker": "AAA"},
        {"published_at": "2024-01-13T12:00:00+01:00", "source": "s", "title": "d", "text": "",
         "url": "u4", "ticker": "AAA"},
    ]))
    ids = dict(zip(articles["title"], articles["id"]))
    base = dict(ticker="AAA", categorie="resultats", surprise=0.5, horizon="immediat",
                resume_fr="r", model="m", prompt_version="v1", anonymized=True)
    results = pd.DataFrame([
        {**base, "article_id": ids["a"], "sentiment": 1.0, "pertinence": 1.0, "confiance": 0.5},
        {**base, "article_id": ids["b"], "sentiment": -1.0, "pertinence": 0.5, "confiance": 0.5},
        {**base, "article_id": ids["c"], "sentiment": 0.4, "pertinence": 1.0, "confiance": 1.0},
        {**base, "article_id": ids["d"], "sentiment": 0.8, "pertinence": 0.0, "confiance": 1.0},
    ])
    return articles, results


def test_daily_sentiment_weighting_and_trading_days():
    articles, results = _articles_and_results()
    prices = {"AAA": frame(range(100, 115), start="2024-01-01")}
    table = article_table(articles, results, prices)
    assert table["day"].dt.strftime("%d/%m").tolist() == ["08/01", "08/01", "09/01", "15/01"]
    daily = daily_sentiment(table).set_index("day")
    monday = daily.loc[pd.Timestamp("2024-01-08")]
    # weights 0.5 and 0.25 -> (1 x 0.5 - 1 x 0.25) / 0.75
    assert monday["sentiment"] == pytest.approx(1 / 3) and monday["n_articles"] == 2
    assert monday["provider_sentiment"] == pytest.approx(0.4)
    assert daily.loc[pd.Timestamp("2024-01-09"), "sentiment"] == pytest.approx(0.4)
    assert pd.Timestamp("2024-01-15") not in daily.index      # only irrelevant articles


def test_horizon_table_and_cutoff_split():
    rng = np.random.default_rng(0)
    n = 200
    sentiment = rng.uniform(-1, 1, n)
    data = pd.DataFrame({
        "day": pd.bdate_range("2024-01-01", periods=n), "sentiment": sentiment,
        "ar0": 0.03 * sentiment, "ar1": rng.normal(0, 0.01, n),
        "ar5": rng.normal(0, 0.02, n), "ar20": rng.normal(0, 0.04, n)})
    table = horizon_table(data)
    assert table.loc["Jour J", "pearson"] == pytest.approx(1.0)       # same-day reaction
    assert abs(table.loc["J+1", "pearson"]) < 0.2 and table.loc["J+1", "pearson_p"] > 0.05
    before, after = split_by_cutoff(data, date(2024, 3, 29))
    assert len(before) + len(after) == n and before["day"].max() <= pd.Timestamp("2024-03-29")
    everything, nothing = split_by_cutoff(data, None)
    assert len(everything) == n and nothing.empty


def test_category_heatmap_hides_small_samples():
    x = np.linspace(-1, 1, 40)
    articles = pd.DataFrame({
        "categorie": ["resultats"] * 40 + ["dirigeants"] * 10,
        "sentiment": list(x) + list(x[:10])})
    for key in ("0", "1", "5", "20"):
        articles[f"ar{key}"] = articles["sentiment"] * (0.01 if key != "20" else -0.01)
    corr, counts = category_heatmap(articles)
    assert corr.loc["resultats", "J+1"] == pytest.approx(1.0)
    assert corr.loc["resultats", "J+20"] == pytest.approx(-1.0)
    assert np.isnan(corr.loc["dirigeants", "J+1"]) and counts.loc["dirigeants", "J+1"] == 10


def test_event_study_known_path():
    # Flat index. The stock gains 2 % on each positive event day, loses 3 % on negative ones.
    n = 120
    bench = frame([100.0] * n)
    returns = np.zeros(n)
    positives, negatives, edge = [15, 45], [85], 117
    returns[positives] = 0.02
    returns[negatives] = -0.03
    stock = frame(100 * np.cumprod(1 + returns))
    days = stock.index
    daily = pd.DataFrame({
        "ticker": "AAA",
        "day": [days[i] for i in positives + negatives + [edge, 110]],
        "sentiment": [0.8, 0.6, -0.9, 0.9, 0.1]})       # last two: too close to the end, weak
    study, counts = event_study(daily, {"AAA": stock}, bench)
    assert counts == {"positive": 2, "negative": 1}
    assert list(study.index) == list(range(-5, 21))
    assert study.loc[-5:-1, "positive"].abs().max() == pytest.approx(0.0)
    assert study.loc[0:, "positive"].to_numpy() == pytest.approx(0.02)
    assert study.loc[-1, "negative"] == pytest.approx(0.0)
    assert study.loc[20, "negative"] == pytest.approx(-0.03)
