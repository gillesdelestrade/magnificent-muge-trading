"""Tables shown in the interface and exported to CSV (French labels)."""
from __future__ import annotations

import pandas as pd

from ..data.universes import display_name
from ..formatting import fmt_date, fmt_eur, fmt_eur_signed, fmt_number, fmt_pct, fmt_quantity
from ..engine.stats import benchmark_return
from ..storage.db import SimulationRun

SIDE_LABELS = {"buy": "Achat", "sell": "Vente"}
EURO_COLUMNS = ["Cours d'exécution", "Montant brut", "Courtage", "TTF", "Frais", "Montant net",
                "Liquidités après opération", "Valeur globale du portefeuille"]


def journal_table(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per operation, numeric values (formatting is applied at display time)."""
    return pd.DataFrame({
        "Date": trades["date"],
        "Action": trades["ticker"].map(display_name),
        "Opération": trades["side"].map(SIDE_LABELS),
        "Motif": trades["reason"],
        "Quantité": trades["quantity"].astype("float64"),
        "Cours d'exécution": trades["price"].astype("float64"),
        "Montant brut": trades["gross"].astype("float64"),
        "Courtage": trades["brokerage"].astype("float64"),
        "TTF": trades["ttf"].astype("float64"),
        "Frais": trades["fees"].astype("float64"),
        "Montant net": trades["net"].astype("float64"),
        "Plus/moins-value (€)": trades["pnl"].astype("float64"),
        "Plus/moins-value (%)": trades["pnl_pct"].astype("float64") * 100.0,
        "Liquidités après opération": trades["cash_after"].astype("float64"),
        "Valeur globale du portefeuille": trades["value_after"].astype("float64"),
    })


def style_journal(table: pd.DataFrame):
    formats = {col: fmt_eur for col in EURO_COLUMNS}
    formats["Date"] = fmt_date
    formats["Quantité"] = fmt_quantity
    formats["Plus/moins-value (€)"] = lambda v: "" if pd.isna(v) else fmt_eur_signed(v)
    formats["Plus/moins-value (%)"] = lambda v: "" if pd.isna(v) else fmt_pct(v / 100, signed=True)

    def pnl_color(v):
        if pd.isna(v) or v == 0:
            return ""
        return "color: #2ECC8F" if v > 0 else "color: #E5484D"

    def side_color(v):
        return "color: #2ECC8F" if v == "Achat" else "color: #E5484D"

    return (table.style.format(formats)
            .map(pnl_color, subset=["Plus/moins-value (€)", "Plus/moins-value (%)"])
            .map(side_color, subset=["Opération"]))


def skipped_table(skipped: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": skipped["date"],
        "Action": skipped["ticker"].map(display_name),
        "Signal": skipped["signal"],
        "Raison de la non-exécution": skipped["cause"],
    })


def _stops_text(p) -> str:
    parts = []
    if p.stop_loss_enabled:
        parts.append(f"stop loss {fmt_pct(p.stop_loss_pct, 1)}")
    if p.trailing_stop_enabled:
        parts.append(f"stop suiveur {fmt_pct(p.trailing_stop_pct, 1)}")
    if p.take_profit_enabled:
        parts.append(f"prise de bénéfices {fmt_pct(p.take_profit_pct, 1, signed=True)}")
    return " · ".join(parts) or "aucun"


def _sizing_text(p) -> str:
    if p.sizing_mode == "risk":
        return (f"perte max. {fmt_pct(p.risk_per_trade_pct, 1)} par trade "
                f"(plafond {fmt_pct(p.position_size_pct, 0)})")
    return f"{fmt_pct(p.position_size_pct, 0)} du portefeuille"


def _news_text(p) -> str:
    if not p.news_filter_enabled:
        return "non utilisées"
    text = f"achat si sentiment ≥ {fmt_number(p.news_buy_threshold, 2)} sur {p.news_buy_window} j"
    if p.news_sell_enabled:
        text += f" · vente si ≤ {fmt_number(p.news_sell_threshold, 2)}"
    return text


def summary_rows(run: SimulationRun) -> dict[str, str]:
    """Indicators of a simulation as formatted text (comparison table)."""
    s, p = run.result.stats, run.result.params
    days = s.get("avg_holding_days")
    return {
        "Période": f"{fmt_date(run.start)} → {fmt_date(run.end)}",
        "Univers": f"{run.universe} ({len(run.tickers)} titres)",
        "Source de données": run.source_label,
        "Capital de départ": fmt_eur(p.initial_capital),
        "Valeur finale": fmt_eur(s["final_value"]),
        "Performance totale": f"{fmt_eur_signed(s['total_return'])} ({fmt_pct(s['total_return_pct'], signed=True)})",
        f"Performance de l'indice ({run.benchmark})": fmt_pct(benchmark_return(run.benchmark_curve), signed=True),
        "Nombre d'opérations": str(s["n_operations"]),
        "Trades gagnants / perdants": f"{s['n_wins']} / {s['n_losses']}",
        "Taux de réussite": fmt_pct(s["win_rate"], 1),
        "Gain moyen": fmt_eur_signed(s["avg_gain"]),
        "Perte moyenne": fmt_eur_signed(s["avg_loss"]),
        "Stops": _stops_text(p),
        "Taille d'une position": _sizing_text(p),
        "Actualités (IA)": _news_text(p),
        "Plateforme": run.result.fee_profile.name,
        "Total des frais": f"{fmt_eur(s['total_fees'])} ({fmt_pct(s['fees_pct_capital'])} du capital)",
        "Perte maximale depuis un sommet": fmt_pct(s["max_drawdown"]),
        "Durée moyenne de détention": "—" if days is None else f"{fmt_number(days, 1)} jours",
    }
