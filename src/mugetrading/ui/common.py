"""Helpers shared by the Streamlit pages."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from ..data.cache import PriceCache, warmup_start
from ..data.provider import DataProvider
from ..data.registry import DEFAULT_PROVIDER_ID, get_provider
from ..engine.fees import FeeProfile, load_profiles
from ..storage.db import Database

MAX_PERIOD = 250            # longest indicator period allowed in the interface (trading days)
DEFAULT_CAPITAL = 1000.0

SETTING_SOURCE = "data_source"
SETTING_CAPITAL = "default_capital"
SETTING_PROFILE = "default_fee_profile"


@st.cache_resource
def get_db() -> Database:
    return Database()


def current_provider() -> DataProvider:
    return get_provider(get_db().get_setting(SETTING_SOURCE, DEFAULT_PROVIDER_ID))


def default_capital() -> float:
    return float(get_db().get_setting(SETTING_CAPITAL, DEFAULT_CAPITAL))


def fee_profiles() -> list[FeeProfile]:
    return load_profiles()


def default_profile_name(profiles: list[FeeProfile]) -> str:
    name = get_db().get_setting(SETTING_PROFILE, profiles[0].name)
    return name if any(p.name == name for p in profiles) else profiles[0].name


def download_start(start: date) -> date:
    """Start of the downloads: margin for the longest possible indicator period."""
    return warmup_start(start, MAX_PERIOD)


@st.cache_data(show_spinner=False, ttl=3600)
def load_stock(source_id: str, ticker: str, start: date, end: date) -> pd.DataFrame:
    """Prices of one stock (from the cache when possible), empty frame on failure."""
    try:
        return PriceCache(get_provider(source_id)).get(ticker, start, end)
    except Exception:
        return pd.DataFrame()


def go_to(page_key: str) -> None:
    pages = st.session_state.get("pages", {})
    if page_key in pages:
        st.switch_page(pages[page_key])


def page_link(page_key: str, label: str) -> None:
    pages = st.session_state.get("pages", {})
    if page_key in pages:
        st.page_link(pages[page_key], label=label)


# -- News & AI settings ---------------------------------------------------------------

SETTING_NEWS_SOURCE = "news_source"
SETTING_AI_MODEL = "ai_model"
SETTING_AI_MAX_COST = "ai_max_cost"
SETTING_AI_ANONYMIZE = "ai_anonymize"
DEFAULT_AI_MAX_COST = 2.0      # dollars per analysis


def current_news_provider():
    from ..news.provider import DEFAULT_NEWS_PROVIDER_ID, get_news_provider
    return get_news_provider(get_db().get_setting(SETTING_NEWS_SOURCE, DEFAULT_NEWS_PROVIDER_ID))


def ai_models():
    """(models, id of the model selected in the settings)."""
    from ..ai.models import get_model, load_models
    models, default = load_models()
    return models, get_model(get_db().get_setting(SETTING_AI_MODEL), models, default).id


def ai_max_cost() -> float:
    return float(get_db().get_setting(SETTING_AI_MAX_COST, DEFAULT_AI_MAX_COST))


def ai_anonymize_default() -> bool:
    return bool(get_db().get_setting(SETTING_AI_ANONYMIZE, True))
