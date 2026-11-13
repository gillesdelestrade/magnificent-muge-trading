from datetime import date

import numpy as np
import pandas as pd

from mugetrading.data.cache import PriceCache, download_universe, warmup_start
from mugetrading.data.provider import DataProvider, DataProviderError, OHLCV_COLUMNS, adjusted_ohlc


class FakeProvider(DataProvider):
    id = "fake"
    label = "Fake"

    def __init__(self, adj_factor=1.0):
        self.calls = []
        self.adj_factor = adj_factor

    def fetch_daily(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        if ticker == "BAD":
            raise DataProviderError("unknown ticker")
        idx = pd.bdate_range(start, end, name="date")
        base = np.array([100.0 + d.toordinal() % 50 for d in idx])
        return pd.DataFrame(
            {"open": base, "high": base + 1, "low": base - 1, "close": base,
             "adj_close": base * self.adj_factor, "volume": 1000.0},
            index=idx,
        )[OHLCV_COLUMNS]


TODAY = date(2025, 6, 30)


def test_cache_downloads_only_missing_days(tmp_path):
    provider = FakeProvider()
    cache = PriceCache(provider, root=tmp_path, today=TODAY)
    df1 = cache.get("AAA", date(2025, 1, 6), date(2025, 1, 31))
    assert len(provider.calls) == 1 and len(df1) == 20

    cache.get("AAA", date(2025, 1, 10), date(2025, 1, 20))  # fully cached
    assert len(provider.calls) == 1

    df3 = cache.get("AAA", date(2025, 1, 6), date(2025, 2, 14))  # only the end is missing
    assert len(provider.calls) == 2
    assert provider.calls[-1][1] >= date(2025, 1, 24)  # not the whole range again
    assert df3.index.is_monotonic_increasing and not df3.index.has_duplicates
    assert len(df3) == 30
    assert (tmp_path / "fake" / "AAA.parquet").exists()  # one folder per source


def test_cache_refreshes_everything_when_adjustment_changed(tmp_path):
    provider = FakeProvider()
    cache = PriceCache(provider, root=tmp_path, today=TODAY)
    cache.get("AAA", date(2025, 1, 6), date(2025, 1, 31))
    provider.adj_factor = 0.9  # a dividend was paid: all adjusted prices changed
    df = cache.get("AAA", date(2025, 1, 6), date(2025, 2, 14))
    assert np.allclose(df["adj_close"], df["close"] * 0.9)


def test_today_is_never_marked_as_covered(tmp_path):
    provider = FakeProvider()
    cache = PriceCache(provider, root=tmp_path, today=date(2025, 1, 15))
    cache.get("AAA", date(2025, 1, 6), date(2025, 1, 31))
    assert provider.calls[-1][2] == date(2025, 1, 14)


def test_download_universe_reports_failures(tmp_path):
    cache = PriceCache(FakeProvider(), root=tmp_path, today=TODAY)
    seen = []
    report = download_universe(cache, ["AAA", "BAD"], date(2025, 1, 6), date(2025, 1, 31),
                               progress=lambda i, n, t: seen.append(i))
    assert list(report.prices) == ["AAA"]
    assert "BAD" in report.failed
    assert report.n_days == 20 and seen[-1] == 2


def test_warmup_covers_longest_period():
    start = date(2025, 3, 3)
    w = warmup_start(start, 20)
    assert len(pd.bdate_range(w, start)) - 1 >= 30


def test_adjusted_ohlc_scales_all_prices():
    idx = pd.bdate_range("2025-01-06", periods=2, name="date")
    df = pd.DataFrame({"open": [10.0, 20], "high": [12.0, 22], "low": [8.0, 18], "close": [10.0, 20],
                       "adj_close": [5.0, 20], "volume": [1.0, 1]}, index=idx)
    adj = adjusted_ohlc(df)
    assert adj["open"].tolist() == [5.0, 20.0] and adj["high"].tolist() == [6.0, 22.0]
