"""Summary statistics and benchmark curve."""
from __future__ import annotations

import pandas as pd


def drawdown(values: pd.Series) -> pd.Series:
    """Loss from the previous peak, as a negative fraction."""
    if values.empty:
        return values
    return values / values.cummax() - 1.0


def benchmark_curve(benchmark: pd.DataFrame | None, index: pd.DatetimeIndex,
                    initial_capital: float) -> pd.Series:
    """Benchmark scaled to the same initial capital, aligned on the simulation days."""
    if benchmark is None or benchmark.empty or len(index) == 0:
        return pd.Series(dtype="float64")
    column = "adj_close" if "adj_close" in benchmark.columns else "close"
    close = benchmark[column].reindex(benchmark.index.union(index)).ffill().reindex(index).bfill()
    if close.isna().all():
        return pd.Series(dtype="float64")
    return initial_capital * close / close.iloc[0]


def compute_stats(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict:
    final_value = float(equity["value"].iloc[-1]) if len(equity) else float(initial_capital)
    sells = trades[trades["side"] == "sell"]
    pnl = sells["pnl"].astype("float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    total_fees = float(trades["fees"].sum()) if len(trades) else 0.0
    dd = drawdown(equity["value"]) if len(equity) else pd.Series(dtype="float64")
    return {
        "initial_capital": float(initial_capital),
        "final_value": final_value,
        "total_return": final_value - initial_capital,
        "total_return_pct": final_value / initial_capital - 1.0,
        "n_operations": int(len(trades)),
        "n_buys": int((trades["side"] == "buy").sum()),
        "n_sells": int(len(sells)),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": float(len(wins) / len(sells)) if len(sells) else None,
        "avg_gain": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "total_fees": total_fees,
        "fees_pct_capital": total_fees / initial_capital,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "avg_holding_days": float(sells["holding_days"].mean()) if len(sells) else None,
        "start": equity.index[0].date().isoformat() if len(equity) else None,
        "end": equity.index[-1].date().isoformat() if len(equity) else None,
    }


def benchmark_return(curve: pd.Series) -> float | None:
    if curve.empty or curve.iloc[0] == 0:
        return None
    return float(curve.iloc[-1] / curve.iloc[0] - 1.0)
