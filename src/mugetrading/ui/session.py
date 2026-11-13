"""Work in progress kept between two launches: the last dataset, the last simulation and
the last set of articles are remembered (SQLite settings) and rebuilt from the local caches,
without any download."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from ..data.cache import PriceCache, download_universe
from ..data.registry import get_provider
from ..news.cache import NewsCache, collect_news
from ..storage.db import SimulationRun
from .common import download_start, get_db

SETTING_LAST_DATASET = "last_dataset"
SETTING_LAST_RUN = "last_run_id"
SETTING_LAST_NEWS = "last_news"
AUTO_HISTORY_SIZE = 20       # automatic simulations kept (named ones are kept for good)


# -- dataset ----------------------------------------------------------------------------

def build_dataset(provider, requested: list[str], universe: str, start: date, end: date,
                  benchmark: str, progress=None, offline: bool = False) -> dict:
    """Prices of the universe (with the margin needed by the indicators) and the benchmark."""
    cache = PriceCache(provider)
    dl_start = download_start(start)
    report = download_universe(cache, requested, dl_start, end, progress, offline=offline)
    try:
        bench_df = (cache.get_cached(benchmark, dl_start, end) if offline
                    else cache.get(benchmark, dl_start, end))
    except Exception:
        bench_df = pd.DataFrame()

    days = pd.DatetimeIndex([])
    no_data_in_period = {}
    for ticker, df in list(report.prices.items()):
        in_period = df.loc[pd.Timestamp(start):]
        if in_period.empty:
            no_data_in_period[ticker] = "Aucune donnée sur la période"
            del report.prices[ticker]
        days = days.union(in_period.index)
    return {
        "source_id": provider.id, "source_label": provider.label, "universe": universe,
        "requested": list(requested), "tickers": list(report.prices), "start": start, "end": end,
        "benchmark": benchmark, "prices": report.prices, "benchmark_df": bench_df,
        "n_days": len(days), "failed": {**report.failed, **no_data_in_period},
        "partial": report.partial,
    }


def remember_dataset(dataset: dict) -> None:
    get_db().set_setting(SETTING_LAST_DATASET, {
        "source_id": dataset["source_id"], "universe": dataset["universe"],
        "requested": dataset["requested"], "start": dataset["start"].isoformat(),
        "end": dataset["end"].isoformat(), "benchmark": dataset["benchmark"]})


def _restore_dataset() -> None:
    saved = get_db().get_setting(SETTING_LAST_DATASET)
    if not saved:
        return
    dataset = build_dataset(
        get_provider(saved["source_id"]), saved["requested"], saved["universe"],
        date.fromisoformat(saved["start"]), date.fromisoformat(saved["end"]),
        saved["benchmark"], offline=True)
    if dataset["tickers"]:
        dataset["restored"] = True
        st.session_state["dataset"] = dataset


# -- simulation -------------------------------------------------------------------------

def set_current_run(run: SimulationRun) -> None:
    st.session_state["run"] = run
    if run.id is not None:
        get_db().set_setting(SETTING_LAST_RUN, run.id)


def autosave_run(run: SimulationRun) -> SimulationRun:
    """Every simulation launched is kept in the automatic history."""
    db = get_db()
    now = datetime.now()
    name = (f"{run.universe} · {run.result.fee_profile.name} · "
            f"{now:%d/%m/%Y} à {now:%H:%M:%S}")
    run.id = db.save_simulation(name, run, auto=True)
    run.name, run.created_at, run.auto = name, now.isoformat(timespec="seconds"), True
    db.prune_auto_simulations(AUTO_HISTORY_SIZE)
    return run


def _restore_run() -> None:
    sim_id = get_db().get_setting(SETTING_LAST_RUN)
    if sim_id is not None:
        st.session_state["run"] = get_db().load_simulation(int(sim_id))


# -- news -------------------------------------------------------------------------------

def remember_news(provider_id: str, dataset: dict) -> None:
    get_db().set_setting(SETTING_LAST_NEWS, {
        "source": provider_id, "tickers": dataset["tickers"],
        "start": dataset["start"].isoformat(), "end": dataset["end"].isoformat()})


def restore_news(provider, dataset: dict) -> None:
    """Articles collected last time for the same titles and period, read from the local cache."""
    if "news" in st.session_state:
        return
    saved = get_db().get_setting(SETTING_LAST_NEWS)
    if (not saved or saved["source"] != provider.id or saved["tickers"] != dataset["tickers"]
            or saved["start"] != dataset["start"].isoformat()
            or saved["end"] != dataset["end"].isoformat()):
        return
    try:
        report = collect_news(NewsCache(provider), dataset["tickers"], dataset["start"],
                              dataset["end"], offline=True)
    except Exception:
        return
    if not report.articles.empty:
        st.session_state["news"] = {"articles": report.articles, "failed": report.failed,
                                    "source": provider.label, "restored": True}


# -- startup ----------------------------------------------------------------------------

def restore_session() -> None:
    """Called once per browser session, before the pages are drawn. A problem with the saved
    state must never prevent the application from starting."""
    if st.session_state.get("_session_restored"):
        return
    st.session_state["_session_restored"] = True
    for key, restore in (("dataset", _restore_dataset), ("run", _restore_run)):
        if key not in st.session_state:
            try:
                restore()
            except Exception:
                pass
