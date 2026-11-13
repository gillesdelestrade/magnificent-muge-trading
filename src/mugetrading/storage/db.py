"""SQLite storage: settings and saved simulations."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..engine.backtest import BacktestResult
from ..engine.fees import FeeProfile
from ..paths import DB_PATH
from ..strategy.params import StrategyParams

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data_source TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    fee_profile_name TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


@dataclass
class SimulationRun:
    """A backtest result with everything needed to display and reproduce it."""
    result: BacktestResult
    source_id: str
    source_label: str
    tickers: list[str]
    universe: str
    start: date
    end: date
    benchmark: str
    benchmark_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    name: str | None = None
    id: int | None = None
    created_at: str | None = None
    auto: bool = False          # kept automatically (not named by the user yet)


def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso", double_precision=15)


def _df_from_json(text: str, date_columns: tuple[str, ...] = (), date_index: bool = False) -> pd.DataFrame:
    raw = json.loads(text)
    df = pd.DataFrame(raw["data"], index=raw["index"], columns=raw["columns"])
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    if date_index:
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.astype("float64")
    return df


class Database:
    def __init__(self, path: Path | None = None):
        self.path = path or DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)
            columns = [row[1] for row in con.execute("PRAGMA table_info(simulations)")]
            if "auto" not in columns:      # databases created before the automatic history
                con.execute("ALTER TABLE simulations ADD COLUMN auto INTEGER NOT NULL DEFAULT 0")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # -- settings --------------------------------------------------------
    def get_setting(self, key: str, default=None):
        with closing(self._connect()) as con:
            row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value) -> None:
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    # -- simulations -----------------------------------------------------
    def save_simulation(self, name: str, run: SimulationRun, auto: bool = False) -> int:
        r = run.result
        payload = {
            "params": r.params.to_dict(),
            "fee_profile": r.fee_profile.to_dict(),   # copy of the values, not only the name
            "stats": r.stats,
            "trades": _df_to_json(r.trades),
            "equity": _df_to_json(r.equity),
            "skipped": _df_to_json(r.skipped),
            "stops": _df_to_json(r.stops),
            "open_positions": r.open_positions,
            "source_label": run.source_label,
            "tickers": run.tickers,
            "universe": run.universe,
            "start": run.start.isoformat(),
            "end": run.end.isoformat(),
            "benchmark": run.benchmark,
            "benchmark_curve": _df_to_json(run.benchmark_curve.to_frame("value")),
        }
        with closing(self._connect()) as con, con:
            cur = con.execute(
                "INSERT INTO simulations (name, created_at, data_source, initial_capital, "
                "fee_profile_name, payload, auto) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, datetime.now().isoformat(timespec="seconds"), run.source_id,
                 r.params.initial_capital, r.fee_profile.name, json.dumps(payload), int(auto)),
            )
            return int(cur.lastrowid)

    def name_simulation(self, sim_id: int, name: str) -> None:
        """Give a name to a simulation: it leaves the automatic history and is kept for good."""
        with closing(self._connect()) as con, con:
            con.execute("UPDATE simulations SET name = ?, auto = 0 WHERE id = ?", (name, sim_id))

    def prune_auto_simulations(self, keep: int) -> None:
        """Keep only the ``keep`` most recent automatic simulations (named ones are never removed)."""
        with closing(self._connect()) as con, con:
            con.execute(
                "DELETE FROM simulations WHERE auto = 1 AND id NOT IN "
                "(SELECT id FROM simulations WHERE auto = 1 ORDER BY id DESC LIMIT ?)", (keep,))

    def list_simulations(self) -> pd.DataFrame:
        with closing(self._connect()) as con:
            return pd.read_sql_query(
                "SELECT id, name, created_at, data_source, initial_capital, fee_profile_name, "
                "auto FROM simulations ORDER BY created_at DESC, id DESC", con)

    def load_simulation(self, sim_id: int) -> SimulationRun:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT id, name, created_at, data_source, payload, auto FROM simulations "
                "WHERE id = ?",
                (sim_id,)).fetchone()
        if row is None:
            raise KeyError(sim_id)
        p = json.loads(row[4])
        result = BacktestResult(
            trades=_df_from_json(p["trades"], ("date", "entry_date")),
            equity=_df_from_json(p["equity"], date_index=True),
            stats=p["stats"],
            skipped=_df_from_json(p["skipped"], ("date",)),
            stops=_df_from_json(p["stops"], ("date",)),
            params=StrategyParams.from_dict(p["params"]),
            fee_profile=FeeProfile.from_dict(p["fee_profile"]),
            open_positions=p.get("open_positions", {}),
        )
        curve = _df_from_json(p["benchmark_curve"], date_index=True)
        return SimulationRun(
            result=result, source_id=row[3], source_label=p["source_label"], tickers=p["tickers"],
            universe=p["universe"], start=date.fromisoformat(p["start"]),
            end=date.fromisoformat(p["end"]), benchmark=p["benchmark"],
            benchmark_curve=curve["value"] if "value" in curve else pd.Series(dtype="float64"),
            name=row[1], id=row[0], created_at=row[2], auto=bool(row[5]),
        )

    def delete_simulation(self, sim_id: int) -> None:
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
