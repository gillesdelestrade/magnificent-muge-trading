"""Manual CSV import: columns ``date_heure ; ticker ; titre ; texte ; source ; url``."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from .provider import PARIS, NewsProvider, NewsProviderError, empty_articles, normalize_articles

CSV_COLUMNS = ["date_heure", "ticker", "titre", "texte", "source", "url"]


class CsvNews(NewsProvider):
    id = "csv"
    label = "Import manuel CSV"
    description = ("Fichier CSV (séparateur ;) : date_heure ; ticker ; titre ; texte ; source ; url. "
                   "Permet de tester sans abonnement.")
    downloads = False

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return empty_articles()   # nothing to download: articles come from parse_csv()


def _parse_dates(values: pd.Series) -> pd.Series:
    """ISO or French (JJ/MM/AAAA HH:MM) date-times; naive values are Paris local time."""
    def one(text: str) -> pd.Timestamp:
        text = str(text).strip()
        try:
            stamp = pd.Timestamp(text) if text[:4].isdigit() else pd.to_datetime(text, dayfirst=True)
        except Exception as exc:
            raise NewsProviderError(f"Date illisible : « {text} »") from exc
        if pd.isna(stamp):
            raise NewsProviderError("Date manquante dans le fichier.")
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(PARIS, ambiguous=True, nonexistent="shift_forward")
        return stamp.tz_convert("UTC")
    return values.map(one)


def parse_csv(content: bytes | str) -> pd.DataFrame:
    """Parse an imported file into the common article format."""
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig", errors="replace")
    lines = [line for line in content.splitlines() if not line.startswith("#")]
    try:
        raw = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", dtype=str, keep_default_na=False)
    except Exception as exc:
        raise NewsProviderError(f"Fichier CSV illisible : {exc}") from exc
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in ("date_heure", "ticker", "titre") if c not in raw.columns]
    if missing:
        raise NewsProviderError("Colonnes manquantes : " + ", ".join(missing))
    for optional in ("texte", "source", "url"):
        if optional not in raw.columns:
            raw[optional] = ""
    raw = raw[raw["titre"].str.strip() != ""]
    if raw.empty:
        return empty_articles()
    return normalize_articles(pd.DataFrame({
        "published_at": _parse_dates(raw["date_heure"]),
        "source": raw["source"].replace("", "Import CSV"),
        "title": raw["titre"], "text": raw["texte"], "url": raw["url"],
        "ticker": raw["ticker"].str.strip().str.upper(),
    }))
