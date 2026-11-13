"""EODHD Financial News API (paid, API key in .env): https://eodhd.com/api/news"""
from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd
import requests

from ..env import get_api_key
from .provider import NewsProvider, NewsProviderError, normalize_articles

API_URL = "https://eodhd.com/api/news"
PAGE_SIZE = 1000


class EodhdNews(NewsProvider):
    id = "eodhd"
    label = "EODHD – Financial News API"
    description = ("Payant (clé API). Articles par titre et par plage de dates, avec un score "
                   "de sentiment fourni par EODHD.")
    needs_api_key = True
    api_key_env = "EODHD_API_KEY"

    def __init__(self, http_get: Callable | None = None, api_key: str | None = None):
        self._get = http_get or requests.get
        self._api_key = api_key

    def to_provider_symbol(self, ticker: str) -> str:
        return ticker.upper()   # EODHD uses the same CODE.EXCHANGE form for Paris (.PA)

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        key = get_api_key(self.api_key_env) if self._api_key is None else self._api_key
        if not key:
            raise NewsProviderError("Clé API EODHD manquante (onglet Paramétrage).")
        rows, offset = [], 0
        while True:
            try:
                response = self._get(API_URL, params={
                    "s": self.to_provider_symbol(ticker), "from": start.isoformat(),
                    "to": end.isoformat(), "limit": PAGE_SIZE, "offset": offset,
                    "api_token": key, "fmt": "json"}, timeout=30)
            except Exception as exc:
                raise NewsProviderError(f"EODHD : {exc}") from exc
            if response.status_code in (401, 403):
                raise NewsProviderError("EODHD : clé API refusée ou abonnement insuffisant.")
            if response.status_code != 200:
                raise NewsProviderError(f"EODHD : erreur HTTP {response.status_code}")
            page = response.json()
            if not isinstance(page, list):
                raise NewsProviderError("EODHD : réponse inattendue.")
            for item in page:
                sentiment = item.get("sentiment") or {}
                rows.append({
                    "published_at": item.get("date"), "source": "EODHD",
                    "title": item.get("title") or "", "text": item.get("content") or "",
                    "url": item.get("link") or "", "ticker": ticker,
                    "provider_sentiment": sentiment.get("polarity") if isinstance(sentiment, dict) else None,
                })
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return normalize_articles(pd.DataFrame(rows))
