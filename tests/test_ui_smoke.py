"""Smoke tests: every page renders without exception (network-free)."""
from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import flat, make_prices


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    from mugetrading.storage import db
    from mugetrading.ui import common
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.sqlite")
    common.get_db.clear()
    yield
    common.get_db.clear()


def _dataset():
    rows = flat(30) + [(100, 108, 100, 107), (108, 112, 107, 111), (111, 113, 96, 97),
                       (97, 98, 95, 96)] + flat(5, 96)
    df = make_prices(rows)
    df["adj_close"] = df["close"]
    return {
        "source_id": "yahoo", "source_label": "Yahoo Finance", "universe": "Test",
        "tickers": ["AAA.PA"], "start": df.index[25].date(), "end": df.index[-1].date(),
        "benchmark": "^FCHI", "prices": {"AAA.PA": df}, "benchmark_df": df, "n_days": 14,
        "failed": {"BAD": "Aucune donnée"}, "partial": {},
    }


def _page(module: str) -> AppTest:
    script = f"from mugetrading.ui import {module}\n{module}.render()\n"
    return AppTest.from_string(script, default_timeout=30)


@pytest.mark.parametrize("module", ["page_home", "page_settings", "page_data", "page_simulation",
                                    "page_results", "page_compare", "page_news_ai"])
def test_pages_render_empty(module):
    at = _page(module).run()
    assert not at.exception, at.exception


def test_full_flow_simulation_results_save_compare(monkeypatch):
    from mugetrading.ui import common
    monkeypatch.setattr(common, "load_stock", lambda *a, **k: _dataset()["prices"]["AAA.PA"])
    from mugetrading.ui import page_results
    monkeypatch.setattr(page_results, "load_stock", common.load_stock)

    at = _page("page_data")
    at.session_state["dataset"] = _dataset()
    at.run()
    assert not at.exception

    at = _page("page_simulation")
    at.session_state["dataset"] = _dataset()
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Comparer les plateformes").click().run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Lancer la simulation").click().run()
    assert not at.exception, at.exception
    run = at.session_state["run"]
    assert len(run.result.trades) == 2      # one buy, one stop
    assert run.auto and run.id is not None  # kept automatically as soon as it is launched

    res = _page("page_results")
    res.session_state["run"] = run
    res.run()
    assert not res.exception, res.exception
    res.text_input[0].set_value("Essai")
    next(b for b in res.button if b.label == "Sauvegarder").click().run()
    assert not res.exception, res.exception
    assert res.session_state["run"].id == run.id and not res.session_state["run"].auto
    assert res.session_state["run"].name == "Essai"

    common.get_db().save_simulation("Essai 2", run)
    cmp = _page("page_compare").run()
    assert not cmp.exception, cmp.exception
    ids = common.get_db().list_simulations()["id"].tolist()
    cmp.multiselect[0].set_value(ids).run()
    assert not cmp.exception, cmp.exception


def test_news_ai_flow_with_fake_model(tmp_path, monkeypatch):
    """Articles -> fake AI analysis -> correlation screen, without any network call."""
    import numpy as np
    from mugetrading.ai import pipeline
    from mugetrading.ai.analyzer import AnalyzerResult, SentimentAnalyzer
    from mugetrading.ai.models import AIModel
    from mugetrading.ai.schema import TickerAnalysis
    from mugetrading.news.provider import normalize_articles
    from mugetrading.ui import page_news_ai

    monkeypatch.setattr(pipeline, "AI_CACHE_PATH", tmp_path / "ai.sqlite")
    page_news_ai._analysis_cache.clear()

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=140, name="date")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "adj_close": close, "volume": 1000.0}, index=idx)
    dataset = {**_dataset(), "tickers": ["MC.PA"], "prices": {"MC.PA": df}, "benchmark_df": df * 1.0,
               "start": idx[0].date(), "end": idx[-1].date()}
    articles = normalize_articles(pd.DataFrame([{
        "published_at": f"{d:%Y-%m-%d}T09:00:00+01:00", "source": "s", "title": f"LVMH jour {i}",
        "text": "Texte", "url": f"https://x/{i}", "ticker": "MC.PA", "provider_sentiment": 0.1}
        for i, d in enumerate(idx[10:110])]))

    class Fake(SentimentAnalyzer):
        model_id = "claude-haiku-5-5"

        def analyze(self, system, user):
            assert "LVMH" not in user          # anonymised by default
            score = (hash(user) % 200) / 100 - 1
            return AnalyzerResult([TickerAnalysis(
                ticker="A", sentiment=score, pertinence=0.9, categorie="resultats", surprise=0.5,
                horizon="immediat", confiance=0.8, resume_fr="Résumé.")], 500, 100)

    monkeypatch.setattr(AIModel, "make_analyzer", lambda self: Fake())

    at = _page("page_news_ai")
    at.session_state["dataset"] = dataset
    at.session_state["news"] = {"articles": articles, "failed": {}, "source": "test"}
    at.run()
    assert not at.exception, at.exception
    next(c for c in at.checkbox if c.label.startswith("Je confirme")).check().run()
    next(b for b in at.button if b.label == "Lancer l'analyse IA").click().run()
    assert not at.exception, at.exception
    report = at.session_state["news_report"]
    assert report.analyzed == 100 and not report.failed
    # Correlation tab now has data: the tables render without error
    assert any("Sentiment IA et rendements" in h.value for h in at.subheader)
    page_news_ai._analysis_cache.clear()


def test_session_is_restored_from_local_caches(tmp_path, monkeypatch):
    """Dataset, last simulation and articles come back after a restart, without any download."""
    from mugetrading.data import cache as price_cache
    from mugetrading.data.provider import DataProvider
    from mugetrading.news import cache as news_cache
    from mugetrading.news.provider import NewsProvider, normalize_articles
    from mugetrading.ui import common, session

    monkeypatch.setattr(price_cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(news_cache, "DATA_DIR", tmp_path)
    data = _dataset()
    prices = data["prices"]["AAA.PA"]

    class Offline(DataProvider):
        id, label, calls = "yahoo", "Yahoo Finance", 0

        def fetch_daily(self, ticker, start, end):
            Offline.calls += 1
            if ticker == "BAD.PA":
                return prices.iloc[:0]
            return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]

    class News(NewsProvider):
        id, label, calls = "eodhd", "EODHD", 0

        def fetch(self, ticker, start, end):
            News.calls += 1
            return normalize_articles(pd.DataFrame([{
                "published_at": f"{data['start']}T09:00:00+01:00", "source": "s", "title": "Titre",
                "text": "Texte", "url": "https://x/1", "ticker": ticker, "provider_sentiment": None}]))

    provider, news_provider = Offline(), News()
    monkeypatch.setattr(session, "get_provider", lambda source_id: provider)

    first = session.build_dataset(provider, ["AAA.PA", "BAD.PA"], "Liste personnalisée",
                                  data["start"], data["end"], "AAA.PA")
    session.remember_dataset(first)
    run = common.get_db().load_simulation(common.get_db().save_simulation("auto", _run_for(first), auto=True))
    common.get_db().set_setting(session.SETTING_LAST_RUN, run.id)
    news_cache.collect_news(news_cache.NewsCache(news_provider), first["tickers"], first["start"], first["end"])
    session.remember_news(news_provider.id, first)
    price_calls, news_calls = Offline.calls, News.calls

    # New browser session: nothing in memory
    script = ("import streamlit as st\nfrom mugetrading.ui import session\n"
              "session.restore_session()\n")
    at = AppTest.from_string(script, default_timeout=30).run()
    assert not at.exception, at.exception
    restored = at.session_state["dataset"]
    assert restored["tickers"] == ["AAA.PA"] and restored["requested"] == ["AAA.PA", "BAD.PA"]
    assert restored["start"] == data["start"] and restored["restored"]
    pd.testing.assert_frame_equal(restored["prices"]["AAA.PA"], first["prices"]["AAA.PA"],
                                  check_like=True, check_freq=False)
    assert at.session_state["run"].id == run.id

    monkeypatch.setattr("streamlit.session_state", {})
    session.restore_news(news_provider, restored)
    import streamlit as st
    assert st.session_state["news"]["articles"]["id"].nunique() == 1
    assert (Offline.calls, News.calls) == (price_calls, news_calls)      # no download


def _run_for(dataset):
    from mugetrading.engine.backtest import run_backtest
    from mugetrading.storage.db import SimulationRun
    from mugetrading.strategy.params import StrategyParams
    result = run_backtest(dataset["prices"], StrategyParams(), start=dataset["start"], end=dataset["end"])
    return SimulationRun(result=result, source_id="yahoo", source_label="Yahoo Finance",
                         tickers=dataset["tickers"], universe=dataset["universe"],
                         start=dataset["start"], end=dataset["end"], benchmark=dataset["benchmark"])
