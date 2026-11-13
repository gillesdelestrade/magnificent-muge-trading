from datetime import date

import pandas as pd
import pytest

from conftest import flat, make_prices
from mugetrading.engine.backtest import run_backtest
from mugetrading.engine.fees import get_profile, load_profiles
from mugetrading.formatting import fmt_eur, fmt_pct, to_french_csv
from mugetrading.storage.db import Database, SimulationRun
from mugetrading.strategy.params import COMBINE_BREAKOUT, StrategyParams


def _run():
    df = make_prices(flat(4) + [(100, 105, 100, 105), (106, 108, 104, 107), (101, 102, 94, 96)])
    params = StrategyParams(momentum_period=3, breakout_period=3, combination=COMBINE_BREAKOUT)
    result = run_backtest({"AAA.PA": df}, params, get_profile("Bourse Direct", load_profiles()))
    curve = pd.Series([1000.0, 1010.0], index=df.index[:2])
    return SimulationRun(result=result, source_id="yahoo", source_label="Yahoo Finance",
                         tickers=["AAA.PA"], universe="Perso", start=date(2024, 1, 1),
                         end=date(2024, 1, 9), benchmark="^FCHI", benchmark_curve=curve)


def test_settings_roundtrip(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    assert db.get_setting("default_capital", 1000.0) == 1000.0
    db.set_setting("default_capital", 2500.5)
    db.set_setting("default_capital", 3000.0)
    assert Database(tmp_path / "db.sqlite").get_setting("default_capital") == 3000.0


def test_simulation_roundtrip(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    run = _run()
    sim_id = db.save_simulation("Essai 1", run)
    listing = db.list_simulations()
    assert listing.iloc[0][["name", "data_source", "fee_profile_name"]].tolist() == [
        "Essai 1", "yahoo", "Bourse Direct"]
    loaded = db.load_simulation(sim_id)
    assert loaded.result.params == run.result.params
    assert loaded.result.fee_profile == run.result.fee_profile     # full copy of the fee grid
    assert loaded.result.stats == run.result.stats
    pd.testing.assert_frame_equal(loaded.result.equity, run.result.equity, check_freq=False)
    assert loaded.result.trades["date"].tolist() == run.result.trades["date"].tolist()
    assert loaded.result.trades["net"].tolist() == pytest.approx(run.result.trades["net"].tolist())
    assert loaded.benchmark_curve.tolist() == [1000.0, 1010.0]
    db.delete_simulation(sim_id)
    assert db.list_simulations().empty


def test_french_formats():
    assert fmt_eur(1234.56) == "1 234,56 €"
    assert fmt_eur(-0.001) == "0,00 €"
    assert fmt_pct(0.0525, signed=True) == "+5,25 %"
    assert fmt_eur(None) == "—"


def test_csv_export_is_excel_french():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-08"]), "Montant": [1234.5]})
    lines = to_french_csv(df, today=date(2026, 11, 11)).decode("utf-8-sig").splitlines()
    assert lines[0] == "# Magnificent Muge Trading — export du 11/11/2026"
    assert lines[1] == "Date;Montant"
    assert lines[2] == "08/01/2024;1234,5000"


def test_automatic_history_is_pruned_but_named_simulations_stay(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    run = _run()
    named = db.save_simulation("À garder", run)
    autos = [db.save_simulation(f"auto {i}", run, auto=True) for i in range(5)]
    db.name_simulation(autos[0], "Finalement nommée")     # leaves the automatic history
    db.prune_auto_simulations(keep=2)
    listing = db.list_simulations()
    assert sorted(listing["id"]) == sorted([named, autos[0], autos[3], autos[4]])
    assert db.load_simulation(autos[0]).auto is False
    assert db.load_simulation(autos[0]).name == "Finalement nommée"
    assert db.load_simulation(autos[4]).auto is True


def test_database_created_before_the_automatic_history_is_migrated(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE simulations (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
                "created_at TEXT NOT NULL, data_source TEXT NOT NULL, initial_capital REAL NOT NULL, "
                "fee_profile_name TEXT NOT NULL, payload TEXT NOT NULL)")
    con.commit()
    con.close()
    db = Database(path)
    sim_id = db.save_simulation("Essai", _run())
    assert db.load_simulation(sim_id).auto is False


def test_backup_contains_databases_news_and_config_but_no_prices_nor_keys(tmp_path, monkeypatch):
    import zipfile
    from datetime import datetime
    from mugetrading import paths
    from mugetrading.storage.backup import create_backup, list_backups

    root = tmp_path / "app"
    (root / "data" / "news" / "eodhd").mkdir(parents=True)
    (root / "data" / "yahoo").mkdir()
    (root / "config").mkdir()
    (root / "data" / "news" / "eodhd" / "MC.PA.parquet").write_bytes(b"news")
    (root / "data" / "yahoo" / "MC.PA.parquet").write_bytes(b"prices")
    (root / "config" / "fee_profiles.yaml").write_text("profiles: []")
    (root / ".env").write_text("ANTHROPIC_API_KEY=secret")
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "DATA_DIR", root / "data")
    monkeypatch.setattr(paths, "CONFIG_DIR", root / "config")
    monkeypatch.setattr(paths, "DB_PATH", root / "data" / "mugetrading.sqlite")
    monkeypatch.setattr(paths, "AI_CACHE_PATH", root / "data" / "ai_cache.sqlite")
    monkeypatch.setattr(paths, "BACKUP_DIR", root / "sauvegardes")
    db = Database(root / "data" / "mugetrading.sqlite")
    db.save_simulation("Essai", _run())

    target = create_backup(now=datetime(2026, 11, 13, 9, 30))
    assert list_backups() == [target]
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        archive.extract("data/mugetrading.sqlite", tmp_path / "restored")
    assert names == {"data/mugetrading.sqlite", "data/news/eodhd/MC.PA.parquet",
                     "config/fee_profiles.yaml"}
    restored = Database(tmp_path / "restored" / "data" / "mugetrading.sqlite")
    assert restored.list_simulations()["name"].tolist() == ["Essai"]
