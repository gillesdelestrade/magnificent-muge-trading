"""Data screen: universe, period, benchmark, download with progress and summary."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from ..data.universes import (BENCHMARKS, DEFAULT_BENCHMARK, display_name, load_universes,
                              parse_tickers)
from ..formatting import fmt_date
from .common import current_provider, page_link
from .session import build_dataset, remember_dataset

CUSTOM = "Liste personnalisée"


def render() -> None:
    st.title("Données")
    provider = current_provider()
    st.caption(f"Source de données : **{provider.label}** (modifiable dans l'onglet Paramétrage) · "
               "cours journaliers, ajustés des dividendes et divisions d'actions.")

    universes = load_universes()
    last = st.session_state.get("dataset") or {}      # the form starts from the last download
    choices = list(universes) + [CUSTOM]
    choice = st.radio("Univers d'actions", choices, horizontal=True,
                      index=choices.index(last["universe"]) if last.get("universe") in choices else 0)
    if choice == CUSTOM:
        text = st.text_area("Tickers au format Yahoo, séparés par des virgules",
                            value=", ".join(last.get("requested", [])) if last.get("universe") == CUSTOM else "",
                            placeholder="MC.PA, AIR.PA, OR.PA",
                            help="Actions de Paris : code suivi de .PA (ex. MC.PA pour LVMH).")
        tickers = parse_tickers(text)
    else:
        tickers = universes[choice]
        st.caption(f"{len(tickers)} titres. Composition actuelle de l'indice (liste dans "
                   "config/universes.yaml) : les sociétés sorties de l'indice n'y figurent pas, "
                   "ce qui embellit un peu les résultats sur le passé.")

    today = date.today()
    col1, col2, col3 = st.columns(3)
    start = col1.date_input("Date de début", value=last.get("start", today - timedelta(days=3 * 365)),
                            max_value=today, format="DD/MM/YYYY")
    end = col2.date_input("Date de fin", value=last.get("end", today), max_value=today, format="DD/MM/YYYY")
    bench_keys = list(BENCHMARKS)
    benchmark = col3.selectbox("Indice de référence", bench_keys,
                               index=bench_keys.index(last["benchmark"] if last.get("benchmark") in bench_keys
                                                     else DEFAULT_BENCHMARK),
                               format_func=lambda k: f"{BENCHMARKS[k]} ({k})")

    if st.button("Télécharger les cours", type="primary"):
        if not tickers:
            st.error("Indiquez au moins un ticker.")
        elif start >= end:
            st.error("La date de début doit précéder la date de fin.")
        else:
            _download(provider, tickers, choice, start, end, benchmark)

    dataset = st.session_state.get("dataset")
    if dataset:
        _summary(dataset)


def _download(provider, tickers, universe, start, end, benchmark) -> None:
    bar = st.progress(0.0, text="Téléchargement…")

    def progress(i: int, n: int, ticker: str) -> None:
        bar.progress(i / max(n, 1), text=f"Téléchargement {i}/{n} — {ticker}" if ticker else "Terminé")

    dataset = build_dataset(provider, tickers, universe, start, end, benchmark, progress)
    bar.empty()
    st.session_state["dataset"] = dataset
    remember_dataset(dataset)      # reloaded from the local cache at the next launch


def _summary(d: dict) -> None:
    st.divider()
    st.subheader("Récapitulatif")
    st.caption(f"{d['universe']} · du {fmt_date(d['start'])} au {fmt_date(d['end'])} · "
               f"source {d['source_label']}")
    if d.get("restored"):
        st.info("Données de votre dernière session, relues depuis le cache local (sans "
                "téléchargement). Cliquez sur « Télécharger les cours » pour les mettre à jour.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titres disponibles", len(d["tickers"]))
    c2.metric("Jours de bourse", d["n_days"])
    c3.metric("Titres en échec", len(d["failed"]))
    c4.metric("Titres avec trous", len(d["partial"]))
    if d["benchmark_df"] is None or d["benchmark_df"].empty:
        st.warning(f"L'indice de référence {d['benchmark']} n'a pas pu être téléchargé : "
                   "la comparaison ne sera pas affichée.")
    if d["failed"]:
        with st.expander(f"Titres en échec ({len(d['failed'])})"):
            st.dataframe(pd.DataFrame({"Action": [display_name(t) for t in d["failed"]],
                                       "Problème": list(d["failed"].values())}),
                         hide_index=True, width="stretch")
    if d["partial"]:
        with st.expander(f"Titres avec trous de données ({len(d['partial'])})"):
            st.caption("Ces titres sont simplement ignorés sur les jours sans cotation.")
            st.dataframe(pd.DataFrame({"Action": [display_name(t) for t in d["partial"]],
                                       "Détail": list(d["partial"].values())}),
                         hide_index=True, width="stretch")
    if d["tickers"]:
        st.success("Données prêtes. Vous pouvez passer à la simulation.")
        page_link("simulation", "Aller à l'écran Simulation →")
    else:
        st.error("Aucun titre n'a pu être téléchargé.")
