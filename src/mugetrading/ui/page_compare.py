"""Saved simulations: reopen, delete, compare 2 to 4 of them."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data.registry import get_provider
from ..formatting import fmt_date, fmt_eur
from . import charts
from .common import get_db, go_to, page_link
from .session import AUTO_HISTORY_SIZE, set_current_run
from .tables import summary_rows


def render() -> None:
    st.title("Comparaison")
    db = get_db()
    listing = db.list_simulations()
    if listing.empty:
        st.info("Aucune simulation pour l'instant. Chaque simulation lancée est conservée ici "
                "automatiquement.")
        page_link("simulation", "Aller à l'écran Simulation →")
        return

    labels = {int(r.id): (r.name if r.auto else f"{r.name} — {fmt_date(r.created_at)}")
              for r in listing.itertuples()}
    st.subheader("Simulations sauvegardées")
    st.caption(f"Les {AUTO_HISTORY_SIZE} dernières simulations lancées sont conservées "
               "automatiquement ; celles auxquelles vous donnez un nom (écran Résultats) sont "
               "gardées définitivement.")
    st.dataframe(pd.DataFrame({
        "Nom": listing["name"],
        "Conservation": listing["auto"].map(lambda a: "Automatique" if a else "Définitive"),
        "Sauvegardée le": listing["created_at"].map(fmt_date),
        "Capital de départ": listing["initial_capital"].map(fmt_eur),
        "Plateforme": listing["fee_profile_name"],
        "Source de données": listing["data_source"].map(lambda s: get_provider(s).label),
    }), hide_index=True, width="stretch")

    col1, col2, col3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    selected = col1.selectbox("Simulation", list(labels), format_func=labels.get)
    if col2.button("Rouvrir", width="stretch"):
        set_current_run(db.load_simulation(selected))
        go_to("results")
    with col3.popover("Supprimer", width="stretch"):
        st.write(f"Supprimer définitivement « {labels[selected]} » ?")
        if st.button("Confirmer la suppression"):
            db.delete_simulation(selected)
            st.rerun()

    st.divider()
    st.subheader("Comparer des simulations")
    chosen = st.multiselect("Choisir 2 à 4 simulations", list(labels), format_func=labels.get,
                            max_selections=4)
    if len(chosen) < 2:
        st.caption("Sélectionnez au moins deux simulations.")
        return
    runs = [db.load_simulation(i) for i in chosen]
    names = []
    for run in runs:   # distinct column names even if two simulations share a name
        name = run.name or f"Simulation {run.id}"
        names.append(name if name not in names else f"{name} (n° {run.id})")
    st.plotly_chart(
        charts.comparison_chart({n: r.result.equity["value"] for n, r in zip(names, runs)}),
        config=charts.PLOTLY_CONFIG)
    st.dataframe(pd.DataFrame({n: summary_rows(r) for n, r in zip(names, runs)}), width="stretch")
