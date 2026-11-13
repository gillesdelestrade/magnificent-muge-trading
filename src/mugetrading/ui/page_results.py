"""Results screen: indicators, charts, journal, CSV export, save."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data.provider import adjusted_ohlc
from ..data.universes import display_name
from ..engine.stats import benchmark_return
from ..formatting import (fmt_date, fmt_eur, fmt_eur_signed, fmt_number, fmt_pct, to_french_csv)
from ..storage.db import SimulationRun
from . import charts
from .common import get_db, load_stock, page_link
from .session import AUTO_HISTORY_SIZE, set_current_run
from .tables import journal_table, skipped_table, style_journal


def _indicators(run: SimulationRun) -> None:
    s = run.result.stats
    bench = benchmark_return(run.benchmark_curve)
    c = st.columns(4)
    c[0].metric("Valeur finale", fmt_eur(s["final_value"]),
                f"{fmt_eur_signed(s['total_return'])} ({fmt_pct(s['total_return_pct'], signed=True)})")
    c[1].metric(f"Indice de référence ({run.benchmark})", fmt_pct(bench, signed=True),
                help="Performance de l'indice sur la même période, ramené au même capital de départ.")
    c[2].metric("Perte maximale depuis un sommet", fmt_pct(s["max_drawdown"]),
                help="Plus forte baisse du portefeuille entre un sommet et le creux qui a suivi.")
    c[3].metric("Total des frais payés", fmt_eur(s["total_fees"]),
                f"{fmt_pct(s['fees_pct_capital'])} du capital de départ", delta_color="off",
                help=f"Plateforme : {run.result.fee_profile.name}")
    c = st.columns(4)
    c[0].metric("Opérations", s["n_operations"],
                f"{s['n_buys']} achats · {s['n_sells']} ventes", delta_color="off")
    c[1].metric("Trades gagnants / perdants", f"{s['n_wins']} / {s['n_losses']}",
                f"Taux de réussite {fmt_pct(s['win_rate'], 1)}", delta_color="off")
    c[2].metric("Gain moyen / perte moyenne",
                f"{fmt_eur_signed(s['avg_gain'])} / {fmt_eur_signed(s['avg_loss'])}")
    days = s.get("avg_holding_days")
    c[3].metric("Durée moyenne de détention",
                "—" if days is None else f"{fmt_number(days, 1)} jours")


def _stock_section(run: SimulationRun) -> None:
    trades = run.result.trades
    traded = sorted(trades["ticker"].unique(), key=display_name)
    if not traded:
        st.info("Aucune opération : pas de graphique par action.")
        return
    col1, col2 = st.columns([3, 1])
    ticker = col1.selectbox("Action", traded, format_func=display_name)
    style = col2.radio("Affichage", ["Chandeliers", "Ligne"], horizontal=True)
    raw = load_stock(run.source_id, ticker, run.start, run.end)
    if raw.empty:
        st.warning("Les cours de cette action ne sont plus disponibles.")
        return
    prices = adjusted_ohlc(raw)
    fig = charts.stock_chart(
        prices, trades[trades["ticker"] == ticker],
        run.result.stops[run.result.stops["ticker"] == ticker],
        display_name(ticker), candles=style == "Chandeliers")
    st.plotly_chart(fig, config=charts.PLOTLY_CONFIG)
    st.caption("Cours ajustés des dividendes et divisions d'actions. Pointillés dorés : niveau "
               "de stop appliqué chaque jour pendant la détention.")


def _journal_section(run: SimulationRun) -> None:
    table = journal_table(run.result.trades)
    if table.empty:
        st.info("Aucune opération sur la période avec ces paramètres.")
        return
    col1, col2, col3 = st.columns(3)
    sides = col1.multiselect("Opération", sorted(table["Opération"].unique()))
    stocks = col2.multiselect("Action", sorted(table["Action"].unique()))
    reasons = col3.multiselect("Motif", sorted(table["Motif"].unique()))
    view = table
    if sides:
        view = view[view["Opération"].isin(sides)]
    if stocks:
        view = view[view["Action"].isin(stocks)]
    if reasons:
        view = view[view["Motif"].isin(reasons)]
    st.dataframe(
        style_journal(view), hide_index=True, width="stretch", height=420,
        column_config={"Frais": st.column_config.Column(
            help="Total des frais de l'opération : courtage de la plateforme + TTF "
                 "(détail dans les deux colonnes précédentes).")})
    st.download_button(
        "Exporter le journal en CSV", to_french_csv(view, "journal des opérations"),
        file_name="muge-trading-journal.csv", mime="text/csv")


def _save_section(run: SimulationRun) -> None:
    if run.id is not None and not run.auto:
        st.caption(f"Simulation sauvegardée sous le nom « {run.name} » le {fmt_date(run.created_at)}.")
        return
    if run.auto:
        st.caption(f"Simulation conservée automatiquement (les {AUTO_HISTORY_SIZE} dernières "
                   "simulations lancées restent disponibles dans l'écran Comparaison). "
                   "Donnez-lui un nom pour la garder définitivement.")
    with st.form("save_simulation"):
        col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
        name = col1.text_input("Nom de la simulation", placeholder="ex. CAC 40 – momentum 20 j – Trade Republic")
        if col2.form_submit_button("Sauvegarder", width="stretch"):
            if not name.strip():
                st.error("Donnez un nom à la simulation.")
            else:
                db = get_db()
                if run.id is not None:
                    db.name_simulation(run.id, name.strip())
                    sim_id = run.id
                else:
                    sim_id = db.save_simulation(name.strip(), run)
                set_current_run(db.load_simulation(sim_id))
                st.rerun()


def render() -> None:
    st.title("Résultats")
    run: SimulationRun | None = st.session_state.get("run")
    if run is None:
        st.info("Aucune simulation à afficher. Lancez-en une, ou rouvrez une simulation sauvegardée.")
        page_link("simulation", "Aller à l'écran Simulation →")
        page_link("compare", "Simulations sauvegardées →")
        return
    r = run.result
    if r.equity.empty:
        st.warning("Aucun jour de bourse sur la période choisie.")
        return
    title = f"**{run.name}** · " if run.name and not run.auto else ""
    st.caption(
        f"{title}{run.universe} ({len(run.tickers)} titres) · du {fmt_date(run.start)} au "
        f"{fmt_date(run.end)} · capital de départ {fmt_eur(r.params.initial_capital)} · "
        f"plateforme {r.fee_profile.name} · source {run.source_label}")

    _indicators(run)
    if r.open_positions:
        held = ", ".join(display_name(t) for t in r.open_positions)
        st.caption(f"Positions encore ouvertes en fin de période, valorisées au dernier cours : {held}.")

    st.subheader("Valeur du portefeuille")
    st.plotly_chart(
        charts.equity_chart(r.equity, run.benchmark_curve, r.trades, run.benchmark),
        config=charts.PLOTLY_CONFIG)
    st.caption("Triangles verts : achats · triangles rouges : ventes. En bas : perte depuis le "
               "dernier sommet (drawdown).")

    tab_journal, tab_stock, tab_alloc, tab_skipped = st.tabs(
        ["Journal des opérations", "Graphique par action", "Répartition",
         f"Signaux non exécutés ({len(r.skipped)})"])
    with tab_journal:
        _journal_section(run)
    with tab_stock:
        _stock_section(run)
    with tab_alloc:
        st.plotly_chart(charts.allocation_chart(r.equity), config=charts.PLOTLY_CONFIG)
        st.caption("Part du portefeuille investie en actions et part restée en liquidités.")
    with tab_skipped:
        if r.skipped.empty:
            st.caption("Tous les signaux d'achat ont été exécutés.")
        else:
            table = skipped_table(r.skipped)
            st.dataframe(table.style.format({"Date": fmt_date}), hide_index=True, width="stretch")
            st.download_button(
                "Exporter en CSV", to_french_csv(table, "signaux non exécutés"),
                file_name="muge-trading-signaux-non-executes.csv", mime="text/csv")

    st.divider()
    _save_section(run)
