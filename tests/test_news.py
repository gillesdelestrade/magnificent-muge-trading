from datetime import date

import pandas as pd
import pytest

from mugetrading.news.cache import NewsCache, collect_news
from mugetrading.news.calendar import assign_trading_days
from mugetrading.news.csv_import import CsvNews, parse_csv
from mugetrading.news.eodhd_news import EodhdNews
from mugetrading.news.provider import (NewsProvider, NewsProviderError, get_news_provider,
                                       list_news_providers, normalize_articles)

# Trading days: Thu 28/03/2024, then Tue 02/04 (Good Friday 29/03 and Easter Monday 01/04 closed)
DAYS = pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-02", "2024-04-03"])


def _assign(*stamps):
    series = pd.Series([pd.Timestamp(s, tz="Europe/Paris") for s in stamps])
    return [d.strftime("%Y-%m-%d") if not pd.isna(d) else None
            for d in assign_trading_days(series, DAYS)]


def test_article_before_close_stays_on_its_day():
    assert _assign("2024-03-27 09:00", "2024-03-27 17:30") == ["2024-03-27", "2024-03-27"]


def test_article_after_close_goes_to_next_trading_day():
    assert _assign("2024-03-27 17:31", "2024-03-27 23:59") == ["2024-03-28", "2024-03-28"]


def test_weekend_and_holidays_go_to_next_trading_day():
    assert _assign("2024-03-28 18:00",    # Thursday evening before the Easter break
                   "2024-03-29 10:00",    # Good Friday
                   "2024-03-30 12:00",    # Saturday
                   "2024-04-01 08:00") == ["2024-04-02"] * 4


def test_article_after_last_known_day_is_unassigned():
    assert _assign("2024-04-03 18:00") == [None]


def test_utc_timestamps_are_converted_to_paris_time():
    # 16:00 UTC in winter = 17:00 in Paris (before the close); 16:45 UTC = 17:45 (after)
    days = pd.DatetimeIndex(["2024-01-09", "2024-01-10"])
    series = pd.Series(pd.to_datetime(["2024-01-09T16:00:00Z", "2024-01-09T16:45:00Z"]))
    assert assign_trading_days(series, days).dt.day.tolist() == [9, 10]


CSV = """# commentaire
date_heure;ticker;titre;texte;source;url
05/01/2024 18:15;mc.pa;LVMH relève ses prévisions;Texte A;Les Echos;https://exemple.fr/a
2024-01-08 09:00;AIR.PA;Airbus décroche une commande;Texte B;;https://exemple.fr/b
"""


def test_csv_import():
    df = parse_csv(CSV.encode("utf-8"))
    assert df["ticker"].tolist() == ["MC.PA", "AIR.PA"]
    assert str(df["published_at"].iloc[0]) == "2024-01-05 18:15:00+01:00"   # naive = Paris time
    assert df["source"].tolist() == ["Les Echos", "Import CSV"]
    assert df["id"].is_unique and df["provider_sentiment"].isna().all()


def test_csv_import_errors():
    with pytest.raises(NewsProviderError, match="Colonnes manquantes"):
        parse_csv("date;ticker\n1;2")
    with pytest.raises(NewsProviderError, match="Date illisible"):
        parse_csv("date_heure;ticker;titre\npas une date;MC.PA;Titre")


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


def test_eodhd_parsing_and_pagination():
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        if params["offset"] == 0:
            return FakeResponse([{"date": "2024-01-05T17:00:00+00:00", "title": f"T{i}",
                                  "content": "C", "link": f"https://x/{i}",
                                  "sentiment": {"polarity": 0.4}} for i in range(1000)])
        return FakeResponse([{"date": "2024-01-06T08:00:00+00:00", "title": "Last",
                              "content": "C", "link": "https://x/last", "sentiment": None}])

    df = EodhdNews(http_get=fake_get, api_key="k").fetch("MC.PA", date(2024, 1, 1), date(2024, 1, 31))
    assert len(df) == 1001 and [c["offset"] for c in calls] == [0, 1000]
    assert calls[0]["s"] == "MC.PA" and calls[0]["from"] == "2024-01-01"
    assert df["provider_sentiment"].iloc[0] == 0.4 and pd.isna(df["provider_sentiment"].iloc[-1])
    assert str(df["published_at"].iloc[0].tz) == "Europe/Paris"


def test_eodhd_errors():
    with pytest.raises(NewsProviderError, match="Clé API"):
        EodhdNews(http_get=lambda *a, **k: None, api_key="").fetch("MC.PA", date(2024, 1, 1), date(2024, 1, 2))
    with pytest.raises(NewsProviderError, match="refusée"):
        EodhdNews(http_get=lambda *a, **k: FakeResponse([], 401), api_key="k").fetch(
            "MC.PA", date(2024, 1, 1), date(2024, 1, 2))


class CountingProvider(NewsProvider):
    id = "fake"
    label = "Fake"

    def __init__(self):
        self.calls = []

    def fetch(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        if ticker == "BAD":
            raise NewsProviderError("boom")
        return normalize_articles(pd.DataFrame([{
            "published_at": f"{d}T10:00:00+00:00", "source": "s", "title": f"{ticker} {d}",
            "text": "", "url": f"https://x/{ticker}/{d}", "ticker": ticker}
            for d in pd.date_range(start, end).strftime("%Y-%m-%d")]))


def test_news_cache_downloads_only_missing_days(tmp_path):
    provider = CountingProvider()
    cache = NewsCache(provider, root=tmp_path, today=date(2024, 6, 1))
    assert len(cache.get("AAA", date(2024, 1, 10), date(2024, 1, 12))) == 3
    assert len(cache.get("AAA", date(2024, 1, 10), date(2024, 1, 11))) == 2
    assert len(provider.calls) == 1                       # second request served by the cache
    assert len(cache.get("AAA", date(2024, 1, 8), date(2024, 1, 14))) == 7
    assert provider.calls[1:] == [("AAA", date(2024, 1, 8), date(2024, 1, 9)),
                                  ("AAA", date(2024, 1, 13), date(2024, 1, 14))]
    assert (tmp_path / "news" / "fake" / "AAA.parquet").exists()   # one folder per source


def test_collect_news_survives_failures(tmp_path):
    cache = NewsCache(CountingProvider(), root=tmp_path, today=date(2024, 6, 1))
    report = collect_news(cache, ["AAA", "BAD"], date(2024, 1, 10), date(2024, 1, 11))
    assert len(report.articles) == 2 and report.failed == {"BAD": "boom"}


def test_csv_source_reads_what_was_imported(tmp_path):
    cache = NewsCache(CsvNews(), root=tmp_path)
    cache.store(parse_csv(CSV))
    cache.store(parse_csv(CSV))    # importing twice does not duplicate
    got = cache.get("MC.PA", date(2024, 1, 1), date(2024, 1, 31))
    assert got["title"].tolist() == ["LVMH relève ses prévisions"]


def test_registry():
    labels = {p.id: p.available for p in list_news_providers()}
    assert labels == {"eodhd": True, "csv": True, "gdelt": False, "rss": False}
    assert get_news_provider("gdelt").id == "eodhd"      # unavailable -> default
    assert get_news_provider("csv").id == "csv"
