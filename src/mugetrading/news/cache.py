"""Local cache of articles, one folder per news source (data/news/<source>/)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from ..paths import DATA_DIR
from .provider import PARIS, NewsProvider, empty_articles, normalize_articles


def _safe(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


class NewsCache:
    def __init__(self, provider: NewsProvider, root: Path | None = None, today: date | None = None):
        self.provider = provider
        self.dir = (root or DATA_DIR) / "news" / provider.id
        self._today = today

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        return self.dir / f"{_safe(ticker)}.parquet", self.dir / f"{_safe(ticker)}.json"

    def read(self, ticker: str) -> pd.DataFrame:
        path, _ = self._paths(ticker)
        if not path.exists():
            return empty_articles()
        try:
            return normalize_articles(pd.read_parquet(path))
        except Exception:
            return empty_articles()

    def _covered(self, ticker: str) -> tuple[date, date] | None:
        _, meta = self._paths(ticker)
        if not meta.exists():
            return None
        try:
            raw = json.loads(meta.read_text())
            return date.fromisoformat(raw["start"]), date.fromisoformat(raw["end"])
        except Exception:
            return None

    def store(self, articles: pd.DataFrame) -> int:
        """Merge imported articles into the cache (used by the CSV import). Returns the row count."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for ticker, group in articles.groupby("ticker"):
            merged = normalize_articles(pd.concat([self.read(ticker), group]))
            merged.to_parquet(self._paths(ticker)[0])
        return len(articles)

    def get(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Articles of ``ticker`` on [start, end]; downloads only what is not cached yet."""
        cached = self.read(ticker)
        if self.provider.downloads:
            today = self._today or date.today()
            cover_end = min(end, today - timedelta(days=1))   # today is still incomplete
            covered = self._covered(ticker)
            if cover_end >= start and (covered is None or start < covered[0] or cover_end > covered[1]):
                pieces = [cached]
                if covered is None:
                    pieces.append(self.provider.fetch(ticker, start, cover_end))
                    new_start, new_end = start, cover_end
                else:
                    if start < covered[0]:
                        pieces.append(self.provider.fetch(ticker, start, covered[0] - timedelta(days=1)))
                    if cover_end > covered[1]:
                        pieces.append(self.provider.fetch(ticker, covered[1] + timedelta(days=1), cover_end))
                    new_start, new_end = min(start, covered[0]), max(cover_end, covered[1])
                cached = normalize_articles(pd.concat(pieces))
                self.dir.mkdir(parents=True, exist_ok=True)
                data_path, meta_path = self._paths(ticker)
                cached.to_parquet(data_path)
                meta_path.write_text(json.dumps({"start": new_start.isoformat(), "end": new_end.isoformat()}))
        return self._window(cached, start, end)

    def get_cached(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Articles already in the cache on [start, end], without any download."""
        return self._window(self.read(ticker), start, end)

    @staticmethod
    def _window(cached: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        lo = pd.Timestamp(start, tz=PARIS)
        hi = pd.Timestamp(end, tz=PARIS) + pd.Timedelta(days=1)
        return cached[(cached["published_at"] >= lo) & (cached["published_at"] < hi)].reset_index(drop=True)


@dataclass
class NewsReport:
    articles: pd.DataFrame = field(default_factory=empty_articles)
    failed: dict[str, str] = field(default_factory=dict)


def collect_news(cache: NewsCache, tickers: list[str], start: date, end: date,
                 progress: Callable[[int, int, str], None] | None = None,
                 offline: bool = False) -> NewsReport:
    """Articles of every ticker; a failing ticker never stops the others.

    ``offline`` reads the cache only (used to restore the previous session).
    """
    report, frames = NewsReport(), []
    for i, ticker in enumerate(tickers):
        if progress:
            progress(i, len(tickers), ticker)
        try:
            frames.append(cache.get_cached(ticker, start, end) if offline
                          else cache.get(ticker, start, end))
        except Exception as exc:
            report.failed[ticker] = str(exc) or exc.__class__.__name__
    if progress:
        progress(len(tickers), len(tickers), "")
    if frames:
        report.articles = normalize_articles(pd.concat(frames))
    return report
