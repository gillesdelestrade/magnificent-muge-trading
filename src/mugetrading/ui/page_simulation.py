"""Simulation screen: capital, strategy parameters, broker fees, launch."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..engine.backtest import run_backtest
from ..engine.fees import FeeProfile
from ..engine.stats import benchmark_curve
from ..formatting import fmt_date, fmt_eur, fmt_number, fmt_pct
from ..storage.db import SimulationRun
from ..strategy.params import (COMBINE_AND, COMBINE_BREAKOUT, COMBINE_MOMENTUM, COMBINE_OR,
                               MIN_CAPITAL, SIZING_PERCENT, SIZING_RISK, StrategyParams)
from .session import analysed_news, analysis_summary, autosave_run, set_current_run
from .common import (MAX_PERIOD, ai_models, current_news_provider, default_capital,
                     default_profile_name, fee_profiles, go_to, page_link)

CAPITAL_SHORTCUTS = [500, 1000, 5000, 10000]
PREVIEW_AMOUNTS = [250, 500, 1000, 2500]
SIZING_LABELS = {
    SIZING_PERCENT: "Part fixe du portefeuille",
    SIZING_RISK: "Selon la perte maximale acceptée par trade",
}
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
    col1, col2, col3 = st.columns(3)
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
    with col3, st.container(border=True):
        use_tp = st.checkbox("Prise de bénéfices", value=False,
                             help="Vendre dès que le gain visé est atteint, sans attendre un stop.")
        tp = st.number_input("Objectif de gain (%)", 0.5, 500.0, 20.0, step=0.5, disabled=not use_tp,
                             help="Vente si le cours monte à prix d'achat × (1 + z %).")
    if use_sl and use_ts:
        st.caption("Les deux stops sont actifs : chaque jour, le niveau le plus élevé des deux s'applique.")
    if use_tp:
        st.caption("Si le stop et l'objectif de gain sont touchés le même jour, la simulation "
                   "retient le stop (hypothèse prudente : les cours journaliers ne disent pas "
                   "lequel a été touché en premier).")
    if not use_sl and not use_ts:
        st.warning("Sans aucun stop, les actions achetées ne sont jamais revendues."
                   if not use_tp else
                   "Sans aucun stop, une action qui baisse n'est jamais revendue.")
    return dict(stop_loss_enabled=use_sl, stop_loss_pct=sl / 100,
                trailing_stop_enabled=use_ts, trailing_stop_pct=ts / 100,
                take_profit_enabled=use_tp, take_profit_pct=tp / 100)


def _portfolio_section(stops: dict, capital: float) -> dict:
    st.subheader("Gestion du portefeuille")
    stop_pcts = [stops[f"{k}_pct"] for k in ("stop_loss", "trailing_stop") if stops[f"{k}_enabled"]]
    mode = st.radio(
        "Calcul de la taille d'une position", [SIZING_PERCENT, SIZING_RISK], horizontal=True,
        format_func=SIZING_LABELS.get,
        help="Part fixe : chaque achat représente le même pourcentage du portefeuille. "
             "Selon la perte maximale : la taille est calculée pour que la perte, si le stop "
             "est touché, ne dépasse pas le pourcentage du portefeuille que vous fixez.")
    risk = 1.0
    if mode == SIZING_RISK and not stop_pcts:
        st.error("Ce calcul a besoin d'un stop : activez le stop loss ou le stop suiveur. "
                 "En attendant, la part fixe est utilisée.")
        mode = SIZING_PERCENT
    col1, col2, col3 = st.columns(3)
    if mode == SIZING_RISK:
        risk = col1.number_input(
            "Perte maximale par trade (% du portefeuille)", 0.1, 20.0, 1.0, step=0.1, format="%.1f",
            help="Perte acceptée sur un trade si le stop est touché, hors frais. Un trou de "
                 "cotation à l'ouverture peut faire perdre davantage.")
    size = col1.slider(
        "Taille maximale d'une position (% du portefeuille)" if mode == SIZING_RISK
        else "Taille d'une position (% du portefeuille)", 5, 100, 25, step=5,
        help="Part de la valeur du portefeuille investie à chaque achat."
        if mode == SIZING_PERCENT else
        "Plafond : même si le calcul selon la perte maximale donne plus, un achat ne "
        "dépasse jamais cette part du portefeuille.")
    if mode == SIZING_RISK:
        stop_pct = min(stop_pcts)
        wanted = risk / 100 / stop_pct
        share = min(wanted, size / 100)
        text = (f"Avec un stop à {fmt_pct(stop_pct, 1)} du prix d'achat, risquer "
                f"{fmt_pct(risk / 100, 1)} revient à investir {fmt_pct(wanted, 1)} du "
                f"portefeuille par achat")
        if wanted > size / 100:
            text += f", ramenés à {fmt_pct(size / 100, 0)} par le plafond"
        st.caption(f"{text} : environ {fmt_eur(capital * share)} au départ.")
    max_pos = col2.number_input("Nombre max de positions simultanées", 1, 50, 4,
                                help="Nombre maximum d'actions différentes détenues en même temps.")
    fractions = col3.checkbox("Autoriser les fractions d'actions", value=False,
                              help="Sinon, nombre entier d'actions, arrondi à l'inférieur.")
    liquidate = col3.checkbox("Vendre les positions en fin de période", value=False,
                              help="Sinon, les positions encore ouvertes sont simplement valorisées "
                                   "au dernier cours de clôture.")
    st.caption("Si plusieurs signaux tombent le même jour, les actions au plus fort momentum passent en premier.")
    return dict(position_size_pct=size / 100, max_positions=int(max_pos),
                allow_fractions=fractions, liquidate_at_end=liquidate,
                sizing_mode=mode, risk_per_trade_pct=risk / 100)


def _news_section(dataset: dict) -> tuple[dict, pd.DataFrame | None]:
    """News filter: decisions drawn from the AI analyses already stored (no AI call here)."""
    st.subheader("Actualités analysées par l'IA")
    enabled = st.checkbox(
        "Tenir compte des actualités analysées par l'IA", value=False,
        help="Chaque jour simulé, seuls les articles déjà parus ce jour-là sont pris en compte : "
             "ils servent à décider à la clôture, et l'ordre est passé à l'ouverture du lendemain. "
             "Un article paru après 17 h 30 compte pour le jour de bourse suivant.")
    off = dict(news_filter_enabled=False)
    if not enabled:
        st.caption("Désactivé : la simulation n'utilise que les cours.")
        return off, None

    summary = analysis_summary()
    if summary.empty:
        st.warning("Aucune analyse IA enregistrée. Récupérez des articles et lancez l'analyse dans "
                   "l'écran Actualités & IA ; la simulation ne lance elle-même aucune analyse.")
        page_link("news", "Aller à l'écran Actualités & IA →")
        return off, None
    models = {m.id: m for m in ai_models()[0]}
    rows = list(summary.itertuples())

    def label(i: int) -> str:
        r = rows[i]
        name = models[r.model].label if r.model in models else r.model
        return (f"{name} · sociétés {'anonymisées' if r.anonymized else 'nommées'} "
                f"({fmt_number(r.n_articles, 0)} articles analysés)")

    choice = rows[st.selectbox("Analyses utilisées", range(len(rows)), format_func=label)]
    with st.spinner("Lecture des actualités analysées…"):
        table = analysed_news(dataset, current_news_provider(), choice.model,
                              bool(choice.anonymized), int(summary["n_articles"].sum()))
    if table.empty:
        st.warning("Aucun article analysé ne concerne ces titres sur cette période. Lancez "
                   "l'analyse dans l'écran Actualités & IA pour ces données.")
        page_link("news", "Aller à l'écran Actualités & IA →")
        return off, None

    first, last = table["day"].min().date(), table["day"].max().date()
    st.caption(f"{fmt_number(table['article_id'].nunique(), 0)} articles analysés sur "
               f"{table['ticker'].nunique()} titres, du {fmt_date(first)} au {fmt_date(last)}. "
               "La simulation relit ces analyses : elle n'appelle pas l'IA et ne coûte rien.")
    if (first - dataset["start"]).days > 10 or (dataset["end"] - last).days > 10:
        st.warning(f"Les actualités analysées ne couvrent que du {fmt_date(first)} au "
                   f"{fmt_date(last)}, alors que la simulation va du {fmt_date(dataset['start'])} "
                   f"au {fmt_date(dataset['end'])}. En dehors de cette plage, il n'y a pas "
                   "d'actualité : le filtre ne vend rien et applique le réglage « sans actualité "
                   "récente » aux achats.")
    cutoff = models[choice.model].knowledge_cutoff if choice.model in models else None
    if cutoff is None or first <= cutoff:
        st.warning("⚠️ Le modèle d'IA a pu « connaître » ce qu'ont fait les actions après ces "
                   "articles" + (f" (ses connaissances vont jusqu'au {fmt_date(cutoff)})" if cutoff
                                 else " (la date de fin de ses connaissances n'est pas renseignée)")
                   + ". Les résultats avec actualités risquent d'être trop beaux : en conditions "
                   "réelles, il ne connaîtra pas la suite.")

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.markdown("**Achats**")
        window = st.number_input(
            "Fenêtre (jours de bourse)", 1, 60, 5,
            help="Nombre de jours de bourse d'actualités pris en compte, jour du signal inclus.")
        threshold = st.slider(
            "Sentiment moyen minimum pour acheter", -1.0, 1.0, 0.0, step=0.05,
            help="De −1 (très négatif) à +1 (très positif). Moyenne des articles de la fenêtre, "
                 "pondérée par leur pertinence et la confiance de l'IA. En dessous du seuil, "
                 "l'achat est abandonné et noté dans les signaux non exécutés.")
        without = st.checkbox("Acheter quand même sans actualité récente", value=True,
                              help="Sinon, une action sans aucun article dans la fenêtre n'est pas achetée.")
    with col2, st.container(border=True):
        st.markdown("**Ventes**")
        sell = st.checkbox("Vendre sur une actualité très négative", value=True,
                           help="Vente à l'ouverture du lendemain, motif « Actualité négative », "
                                "sans attendre le stop.")
        sell_threshold = st.slider("Sentiment déclenchant la vente (≤)", -1.0, 0.0, -0.5, step=0.05,
                                   disabled=not sell)
        pertinence = st.slider("Pertinence minimale de l'article", 0.0, 1.0, 0.7, step=0.05,
                               disabled=not sell,
                               help="De 0 à 1 : l'article parle-t-il vraiment de cette société ?")
    st.caption("Pour mesurer l'effet des actualités, lancez la même simulation avec et sans cette "
               "option, puis comparez-les dans l'écran Comparaison.")
    return dict(news_filter_enabled=True, news_buy_window=int(window), news_buy_threshold=threshold,
                news_buy_without_news=without, news_sell_enabled=sell,
                news_sell_threshold=sell_threshold, news_sell_min_pertinence=pertinence), table


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


def _execute(dataset: dict, params: StrategyParams, profile: FeeProfile,
             news: pd.DataFrame | None = None) -> SimulationRun:
    result = run_backtest(dataset["prices"], params, profile, dataset["start"], dataset["end"], news)
    curve = benchmark_curve(dataset["benchmark_df"], result.equity.index, params.initial_capital)
    return SimulationRun(
        result=result, source_id=dataset["source_id"], source_label=dataset["source_label"],
        tickers=dataset["tickers"], universe=dataset["universe"], start=dataset["start"],
        end=dataset["end"], benchmark=dataset["benchmark"], benchmark_curve=curve)


def _compare_platforms(dataset: dict, params: StrategyParams, profiles: list[FeeProfile],
                       news: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    bar = st.progress(0.0, text="Comparaison des plateformes…")
    for i, profile in enumerate(profiles):
        stats = run_backtest(dataset["prices"], params, profile, dataset["start"], dataset["end"],
                             news).stats
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
    portfolio = _portfolio_section(stops, capital)
    profiles = fee_profiles()
    news_options, news = _news_section(dataset)
    profile, costs = _fees_section(profiles)

    position_amount = capital * portfolio["position_size_pct"]
    if portfolio["sizing_mode"] == SIZING_RISK:
        closest_stop = min(stops[f"{k}_pct"] for k in ("stop_loss", "trailing_stop")
                           if stops[f"{k}_enabled"])
        position_amount = min(position_amount,
                              capital * portfolio["risk_per_trade_pct"] / closest_stop)
    if profile.brokerage(position_amount) > 0.01 * position_amount:
        st.warning(
            "Avec ce capital, les frais vont peser lourd sur chaque opération : "
            f"{fmt_eur(profile.brokerage(position_amount))} pour un ordre de {fmt_eur(position_amount)}, "
            f"soit {fmt_pct(profile.brokerage(position_amount) / position_amount)}.")
    if signals is None:
        return
    params = StrategyParams(initial_capital=capital, **signals, **stops, **portfolio,
                            **news_options, **costs)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Lancer la simulation", type="primary", width="stretch"):
        with st.spinner("Simulation en cours…"):
            set_current_run(autosave_run(_execute(dataset, params, profile, news)))
        st.session_state.pop("platform_comparison", None)
        go_to("results")
    if col2.button("Comparer les plateformes", width="stretch",
                   help="Relance la même simulation avec chaque plateforme : seul le courtier change."):
        st.session_state["platform_comparison"] = _compare_platforms(dataset, params, profiles, news)

    comparison = st.session_state.get("platform_comparison")
    if comparison is not None:
        st.subheader("Comparaison des plateformes")
        st.caption("Même stratégie, mêmes données : seul le courtier change.")
        st.dataframe(
            comparison.style.format({
                "Total des frais payés": fmt_eur, "Valeur finale du portefeuille": fmt_eur,
                "Performance nette": lambda v: fmt_pct(v, signed=True)}),
            hide_index=True, width="stretch")
