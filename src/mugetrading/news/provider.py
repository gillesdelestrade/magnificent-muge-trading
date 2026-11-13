"""NewsProvider interface + registry of news sources.

Articles are handled as DataFrames with one row per (article, ticker):
``id, published_at (tz-aware, Europe/Paris), source, title, text, url, ticker,
provider_sentiment`` (score given by the source, NaN if none).
Full texts stay in the local cache, for personal use only (never republished).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

PARIS = "Europe/Paris"
ARTICLE_COLUMNS = ["id", "published_at", "source", "title", "text", "url", "ticker",
                   "provider_sentiment"]


class NewsProviderError(Exception):
    """Raised when a news source cannot return articles."""


def article_id(url: str, title: str, published_at) -> str:
    basis = url.strip() or f"{title.strip()}|{published_at}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def empty_articles() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in ARTICLE_COLUMNS})
    df["published_at"] = pd.Series(dtype=f"datetime64[ns, {PARIS}]")
    df["provider_sentiment"] = pd.Series(dtype="float64")
    return df


def normalize_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the common format: columns, Paris timezone, no duplicates, sorted."""
    if df is None or df.empty:
        return empty_articles()
    out = df.copy()
    if "provider_sentiment" not in out.columns:
        out["provider_sentiment"] = float("nan")
    stamps = pd.to_datetime(out["published_at"], utc=True).dt.tz_convert(PARIS)
    out["published_at"] = stamps.astype(f"datetime64[ns, {PARIS}]")
    for col in ("source", "title", "text", "url", "ticker"):
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["provider_sentiment"] = pd.to_numeric(out["provider_sentiment"], errors="coerce")
    if "id" not in out.columns:
        out["id"] = [article_id(u, t, p) for u, t, p in
                     zip(out["url"], out["title"], out["published_at"])]
    out = out[ARTICLE_COLUMNS].drop_duplicates(subset=["id", "ticker"], keep="last")
    return out.sort_values(["published_at", "id", "ticker"]).reset_index(drop=True)


class NewsProvider(ABC):
    id: str = ""
    label: str = ""
    description: str = ""
    available: bool = True
    needs_api_key: bool = False
    api_key_env: str | None = None
    downloads: bool = True      # False: articles only come from a manual import

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Articles about ``ticker`` published between ``start`` and ``end`` (inclusive)."""


class _ComingSoon(NewsProvider):
    available = False

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError(f"{self.label} : bientôt disponible")


class GdeltNews(_ComingSoon):
    id = "gdelt"
    label = "GDELT DOC 2.0"
    description = "Gratuit, sans clé, multilingue ; seulement les 3 derniers mois."


class RssCollector(_ComingSoon):
    id = "rss"
    label = "Collecteur RSS maison"
    description = "Archive chaque soir des flux RSS pour constituer notre propre historique."


DEFAULT_NEWS_PROVIDER_ID = "eodhd"


def _providers() -> dict[str, NewsProvider]:
    from .csv_import import CsvNews
    from .eodhd_news import EodhdNews
    return {p.id: p for p in (EodhdNews(), CsvNews(), GdeltNews(), RssCollector())}


def list_news_providers() -> list[NewsProvider]:
    return list(_providers().values())


def get_news_provider(provider_id: str | None = None) -> NewsProvider:
    providers = _providers()
    provider = providers.get(provider_id or DEFAULT_NEWS_PROVIDER_ID)
    if provider is None or not provider.available:
        provider = providers[DEFAULT_NEWS_PROVIDER_ID]
    return provider
