"""Simulation screen: capital, strategy parameters, broker fees, launch."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..engine.backtest import run_backtest
from ..engine.fees import FeeProfile
from ..engine.stats import benchmark_curve
from ..formatting import fmt_date, fmt_eur, fmt_pct
from ..storage.db import SimulationRun
from ..strategy.params import (COMBINE_AND, COMBINE_BREAKOUT, COMBINE_MOMENTUM, COMBINE_OR,
                               MIN_CAPITAL, StrategyParams)
from .session import autosave_run, set_current_run
from .common import MAX_PERIOD, default_capital, default_profile_name, fee_profiles, go_to, page_link

CAPITAL_SHORTCUTS = [500, 1000, 5000, 10000]
PREVIEW_AMOUNTS = [250, 500, 1000, 2500]
COMBINATION_LABELS = {
    COMBINE_AND: "Les deux (ET)",
    COMBINE_OR: "L'un ou l'autre (OU)",
}


def _set_capital(value: float) -> None:
    st.session_state["sim_capital"] = float(value)


def _capital_section() -> float:
    st.subheader("Capital de départ")
    if "sim_capital" not in st.session_state:
        st.session_state["sim_capital"] = default_capital()
    cols = st.columns([2, 1, 1, 1, 1], vertical_alignment="bottom")
    capital = cols[0].number_input(
        "Capital de départ (€)", min_value=MIN_CAPITAL, step=100.0, format="%.2f",
        key="sim_capital",
        help="Liquidités disponibles au premier jour. Minimum 100 €. La valeur par défaut se "
             "règle dans l'onglet Paramétrage.")
    for col, amount in zip(cols[1:], CAPITAL_SHORTCUTS):
        col.button(fmt_eur(amount, 0), on_click=_set_capital, args=(amount,), width="stretch")
    return round(float(capital), 2)


def _signals_section() -> dict | None:
    st.subheader("Signaux d'achat")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        use_mom = st.checkbox("Momentum", value=True,
                              help="Acheter une action qui a nettement monté sur les derniers jours.")
        mom_period = st.number_input(
            "Période (jours de bourse)", 1, MAX_PERIOD, 20, key="mom_period", disabled=not use_mom,
            help="Nombre de jours de bourse sur lesquels on mesure la variation du cours.")
        mom_threshold = st.number_input(
            "Seuil (%)", 0.0, 100.0, 5.0, step=0.5, key="mom_threshold", disabled=not use_mom,
            help="Signal si le cours a monté d'au moins ce pourcentage sur la période.")
    with col2, st.container(border=True):
        use_brk = st.checkbox("Breakout (cassure)", value=True,
                              help="Acheter une action qui dépasse son plus haut récent.")
        brk_period = st.number_input(
            "Période (jours de bourse)", 1, MAX_PERIOD, 20, key="brk_period", disabled=not use_brk,
            help="Signal si la clôture dépasse le plus haut des N jours précédents.")
        brk_margin = st.number_input(
            "Marge (%)", 0.0, 50.0, 0.0, step=0.5, key="brk_margin", disabled=not use_brk,
            help="Dépassement minimum exigé au-dessus de ce plus haut.")
    if use_mom and use_brk:
        combination = st.radio(
            "Combinaison des signaux", list(COMBINATION_LABELS), horizontal=True,
            format_func=COMBINATION_LABELS.get,
            help="ET : les deux signaux le même jour (plus sélectif). OU : un seul suffit.")
    elif use_mom:
        combination = COMBINE_MOMENTUM
    elif use_brk:
        combination = COMBINE_BREAKOUT
    else:
        st.error("Activez au moins un signal d'achat.")
        return None
    return dict(momentum_period=int(mom_period), momentum_threshold=mom_threshold / 100,
                breakout_period=int(brk_period), breakout_margin=brk_margin / 100,
                combination=combination)


def _stops_section() -> dict:
    st.subheader("Signaux de vente")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        use_sl = st.checkbox("Stop loss", value=True,
                             help="Vendre si le cours tombe trop bas par rapport au prix d'achat.")
        sl = st.number_input("Stop loss (%)", 0.5, 90.0, 8.0, step=0.5, disabled=not use_sl,
                             help="Vente si le cours tombe à prix d'achat × (1 − x %).")
    with col2, st.container(border=True):
        use_ts = st.checkbox("Stop suiveur", value=True,
                             help="Le seuil de vente monte avec le cours et ne redescend jamais.")
        ts = st.number_input("Stop suiveur (%)", 0.5, 90.0, 10.0, step=0.5, disabled=not use_ts,
                             help="Vente si le cours tombe à plus haut atteint depuis l'achat × (1 − y %).")
    if use_sl and use_ts:
        st.caption("Les deux stops sont actifs : chaque jour, le niveau le plus élevé des deux s'applique.")
    if not use_sl and not use_ts:
        st.warning("Sans aucun stop, les actions achetées ne sont jamais revendues.")
    return dict(stop_loss_enabled=use_sl, stop_loss_pct=sl / 100,
                trailing_stop_enabled=use_ts, trailing_stop_pct=ts / 100)


def _portfolio_section() -> dict:
    st.subheader("Gestion du portefeuille")
    col1, col2, col3 = st.columns(3)
    size = col1.slider("Taille d'une position (% du portefeuille)", 5, 100, 25, step=5,
                       help="Part de la valeur du portefeuille investie à chaque achat.")
    max_pos = col2.number_input("Nombre max de positions simultanées", 1, 50, 4,
                                help="Nombre maximum d'actions différentes détenues en même temps.")
    fractions = col3.checkbox("Autoriser les fractions d'actions", value=False,
                              help="Sinon, nombre entier d'actions, arrondi à l'inférieur.")
    liquidate = col3.checkbox("Vendre les positions en fin de période", value=False,
                              help="Sinon, les positions encore ouvertes sont simplement valorisées "
                                   "au dernier cours de clôture.")
    st.caption("Si plusieurs signaux tombent le même jour, les actions au plus fort momentum passent en premier.")
    return dict(position_size_pct=size / 100, max_positions=int(max_pos),
                allow_fractions=fractions, liquidate_at_end=liquidate)


def _fees_section(profiles: list[FeeProfile]) -> tuple[FeeProfile, dict]:
    st.subheader("Frais de transaction")
    names = [p.name for p in profiles]
    if st.session_state.get("sim_profile") not in names:
        st.session_state["sim_profile"] = default_profile_name(profiles)
    col1, col2 = st.columns([1, 2])
    name = col1.selectbox("Plateforme (courtier)", names, key="sim_profile",
                          help="Les frais de chaque opération sont calculés selon la grille de ce courtier.")
    profile = profiles[names.index(name)]
    with col2:
        st.caption("Aperçu : coût d'un ordre avec cette plateforme")
        for col, amount in zip(st.columns(len(PREVIEW_AMOUNTS)), PREVIEW_AMOUNTS):
            col.metric(f"Ordre de {fmt_eur(amount, 0)}", fmt_eur(profile.brokerage(amount)))
    note = profile.description
    if profile.verified_on:
        note += f" · Grille vérifiée le {fmt_date(profile.verified_on)}"
    if profile.free_orders_per_month:
        note += " · L'aperçu ne tient pas compte des ordres gratuits du mois"
    st.caption(note)
    if profile.is_stale():
        st.warning("Cette grille de frais a été vérifiée il y a plus de 6 mois : "
                   "pensez à la revérifier dans l'onglet Paramétrage.")
    col1, col2, col3 = st.columns(3)
    ttf_on = col1.checkbox("Taxe sur les transactions financières (TTF)", value=True,
                           help="Taxe française à l'achat. Simplification : appliquée à toutes "
                                "les actions cotées à Paris (.PA).")
    ttf = col2.number_input("Taux de TTF (%)", 0.0, 5.0, 0.4, step=0.1, disabled=not ttf_on)
    slippage = col3.number_input(
        "Glissement (%)", 0.0, 5.0, 0.05, step=0.01, format="%.2f",
        help="Écart défavorable entre le cours théorique et le prix réellement obtenu.")
    return profile, dict(ttf_enabled=ttf_on, ttf_rate=ttf / 100, slippage=slippage / 100)


def _execute(dataset: dict, params: StrategyParams, profile: FeeProfile) -> SimulationRun:
    result = run_backtest(dataset["prices"], params, profile, dataset["start"], dataset["end"])
    curve = benchmark_curve(dataset["benchmark_df"], result.equity.index, params.initial_capital)
    return SimulationRun(
        result=result, source_id=dataset["source_id"], source_label=dataset["source_label"],
        tickers=dataset["tickers"], universe=dataset["universe"], start=dataset["start"],
        end=dataset["end"], benchmark=dataset["benchmark"], benchmark_curve=curve)


def _compare_platforms(dataset: dict, params: StrategyParams, profiles: list[FeeProfile]) -> pd.DataFrame:
    rows = []
    bar = st.progress(0.0, text="Comparaison des plateformes…")
    for i, profile in enumerate(profiles):
        stats = run_backtest(dataset["prices"], params, profile, dataset["start"], dataset["end"]).stats
        rows.append({"Plateforme": profile.name, "Total des frais payés": stats["total_fees"],
                     "Valeur finale du portefeuille": stats["final_value"],
                     "Performance nette": stats["total_return_pct"]})
        bar.progress((i + 1) / len(profiles), text=f"Comparaison des plateformes… {profile.name}")
    bar.empty()
    return pd.DataFrame(rows).sort_values("Valeur finale du portefeuille", ascending=False)


def render() -> None:
    st.title("Simulation")
    dataset = st.session_state.get("dataset")
    if not dataset or not dataset.get("tickers"):
        st.info("Commencez par télécharger des cours dans l'écran Données.")
        page_link("data", "Aller à l'écran Données →")
        return
    st.caption(f"{dataset['universe']} · {len(dataset['tickers'])} titres · du "
               f"{fmt_date(dataset['start'])} au {fmt_date(dataset['end'])} · "
               f"source {dataset['source_label']}")

    capital = _capital_section()
    signals = _signals_section()
    stops = _stops_section()
    portfolio = _portfolio_section()
    profiles = fee_profiles()
    profile, costs = _fees_section(profiles)

    position_amount = capital * portfolio["position_size_pct"]
    if profile.brokerage(position_amount) > 0.01 * position_amount:
        st.warning(
            "Avec ce capital, les frais vont peser lourd sur chaque opération : "
            f"{fmt_eur(profile.brokerage(position_amount))} pour un ordre de {fmt_eur(position_amount)}, "
            f"soit {fmt_pct(profile.brokerage(position_amount) / position_amount)}.")
    if signals is None:
        return
    params = StrategyParams(initial_capital=capital, **signals, **stops, **portfolio, **costs)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Lancer la simulation", type="primary", width="stretch"):
        with st.spinner("Simulation en cours…"):
            set_current_run(autosave_run(_execute(dataset, params, profile)))
        st.session_state.pop("platform_comparison", None)
        go_to("results")
    if col2.button("Comparer les plateformes", width="stretch",
                   help="Relance la même simulation avec chaque plateforme : seul le courtier change."):
        st.session_state["platform_comparison"] = _compare_platforms(dataset, params, profiles)

    comparison = st.session_state.get("platform_comparison")
    if comparison is not None:
        st.subheader("Comparaison des plateformes")
        st.caption("Même stratégie, mêmes données : seul le courtier change.")
        st.dataframe(
            comparison.style.format({
                "Total des frais payés": fmt_eur, "Valeur finale du portefeuille": fmt_eur,
                "Performance nette": lambda v: fmt_pct(v, signed=True)}),
            hide_index=True, width="stretch")
