"""Replace company names, brands and tickers by neutral labels before sending to the AI.

A model trained on data covering the tested period may "know" what the share did after
the article; hiding the identity of the company limits that leak.
"""
from __future__ import annotations

import re
import string
import unicodedata
from functools import lru_cache

import yaml

from ..data.universes import company_name
from ..paths import ALIASES_PATH


@lru_cache(maxsize=1)
def _aliases() -> dict[str, list[str]]:
    if not ALIASES_PATH.exists():
        return {}
    raw = yaml.safe_load(ALIASES_PATH.read_text(encoding="utf-8")) or {}
    return {str(k).upper(): [str(a) for a in (v or [])] for k, v in raw.items()}


def _unaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def company_names(ticker: str) -> list[str]:
    """Official name, aliases and brands of a company (with unaccented variants)."""
    names = []
    official = company_name(ticker)
    if official != ticker:
        names.append(official)
    names += _aliases().get(ticker.upper(), [])
    names += [_unaccent(n) for n in names]
    return list(dict.fromkeys(n for n in names if n.strip()))


def label(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'..."""
    letters = string.ascii_uppercase
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = letters[rem] + out
    return out


def _name_pattern(name: str) -> re.Pattern:
    # Whole words only, case-insensitive, tolerant to straight/curly apostrophes.
    escaped = re.escape(name).replace("'", "['’]").replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)


def anonymize(text: str, tickers: list[str],
              names: dict[str, list[str]] | None = None) -> tuple[str, dict[str, str]]:
    """Returns (anonymized text, mapping label -> ticker).

    Each company becomes « la société A », « la société B »… Its names and brands, its
    full ticker (MC.PA) and its exchange code (MC) are replaced.
    """
    mapping: dict[str, str] = {}
    replacements: list[tuple[re.Pattern, str]] = []
    for i, ticker in enumerate(tickers):
        tag = label(i)
        mapping[tag] = ticker
        placeholder = f"la société {tag}"
        known = names.get(ticker, []) if names is not None else company_names(ticker)
        for name in known:
            replacements.append((_name_pattern(name), placeholder))
        root = ticker.split(".")[0]
        replacements.append((re.compile(rf"(?<![\w.]){re.escape(ticker)}(?![\w])", re.IGNORECASE), placeholder))
        if len(root) >= 3:
            # Exchange code as a whole upper-case word
            replacements.append((re.compile(rf"(?<![\w.]){re.escape(root)}(?![\w.])"), placeholder))
        else:
            # Short codes (AI, OR, CA...) are common words: only replaced in "(AI)" or "EPA:AI"
            replacements.append((re.compile(rf"(?<=[(:])\s*{re.escape(root)}(?=\s*[),])"), placeholder))
    # Longest names first, so "Louis Vuitton" is replaced before "Vuitton"-like substrings
    replacements.sort(key=lambda r: -len(r[0].pattern))
    for pattern, placeholder in replacements:
        text = pattern.sub(placeholder, text)
    return text, mapping
