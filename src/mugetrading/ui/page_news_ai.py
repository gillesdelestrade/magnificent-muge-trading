"""« Actualités & IA » screen: collect news, analyse them with an AI model, then measure
whether the information is correlated with the price moves."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ai.claude_analyzer import API_KEY_ENV
from ..ai.models import AIModel
from ..ai.pipeline import AnalysisCache, build_jobs, estimate_cost, load_results, run_analysis
from ..ai.schema import CATEGORIES
from ..analysis.correlation import (HORIZONS, MIN_OBSERVATIONS, category_heatmap, group_table,
                                    horizon_table, returns_table, split_by_cutoff, strong_days)
from ..analysis.event_study import event_study
from ..analysis.sentiment_daily import article_table, daily_sentiment
from ..data.provider import adjusted_ohlc
from ..data.universes import display_name
from ..env import get_api_key
from ..formatting import fmt_date, fmt_number, fmt_pct, fmt_usd, to_french_csv
from ..news.cache import NewsCache, collect_news
from ..news.csv_import import CSV_COLUMNS, parse_csv
from ..news.provider import NewsProviderError
from . import charts
from .session import remember_news, restore_news
from .common import (ai_anonymize_default, ai_max_cost, ai_models, current_news_provider,
                     page_link)

CSV_TEMPLATE = (";".join(CSV_COLUMNS) + "\n"
                "05/01/2024 18:15;MC.PA;LVMH relève ses prévisions;Texte de l'article…;"
                "Les Echos;https://exemple.fr/article\n")


@st.cache_resource
def _analysis_cache() -> AnalysisCache:
    return AnalysisCache()


# -- 1. Articles ------------------------------------------------------------------------

def _articles_tab(dataset: dict) -> None:
    provider = current_news_provider()
    st.caption(f"Source d'actualités : **{provider.label}** (modifiable dans l'onglet Paramétrage). "
               "Les textes restent dans le cache local, pour un usage personnel.")
    cache = NewsCache(provider)

    if not provider.downloads:
        st.markdown("Fichier CSV, séparateur `;`, colonnes : `" + " ; ".join(CSV_COLUMNS) + "`. "
                    "Dates au format JJ/MM/AAAA HH:MM, heure de Paris.")
        st.download_button("Télécharger un modèle de fichier", CSV_TEMPLATE.encode("utf-8-sig"),
                           file_name="modele-actualites.csv", mime="text/csv")
        uploaded = st.file_uploader("Fichier d'actualités (CSV)", type=["csv"])
        if uploaded is not None and st.button("Importer le fichier", type="primary"):
            try:
                imported = parse_csv(uploaded.getvalue())
            except NewsProviderError as exc:
                st.error(str(exc))
            else:
                cache.store(imported)
                st.success(f"{imported['id'].nunique()} articles importés.")
    elif provider.needs_api_key and not get_api_key(provider.api_key_env):
        st.warning(f"La clé API de {provider.label} n'est pas renseignée (onglet Paramétrage).")

    label = "Charger les articles importés" if not provider.downloads else "Récupérer les actualités"
    if st.button(label, type="primary" if provider.downloads else "secondary"):
        bar = st.progress(0.0, text="Récupération des actualités…")
        report = collect_news(
            cache, dataset["tickers"], dataset["start"], dataset["end"],
            lambda i, n, t: bar.progress(i / max(n, 1), text=f"Actualités {i}/{n} — {t}"))
        bar.empty()
        st.session_state["news"] = {"articles": report.articles, "failed": report.failed,
                                    "source": provider.label}
        remember_news(provider.id, dataset)     # reloaded from the local cache next time

    news = st.session_state.get("news")
    if not news:
        return
    articles = news["articles"]
    if news.get("restored"):
        st.info("Articles de votre dernière session, relus depuis le cache local (sans "
                "téléchargement).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", articles["id"].nunique())
    c2.metric("Titres avec actualités", articles["ticker"].nunique())
    c3.metric("Titres en échec", len(news["failed"]))
    if news["failed"]:
        with st.expander(f"Titres en échec ({len(news['failed'])})"):
            st.dataframe(pd.DataFrame({"Action": [display_name(t) for t in news["failed"]],
                                       "Problème": list(news["failed"].values())}),
                         hide_index=True, width="stretch")
    if articles.empty:
        st.info("Aucun article sur ces titres et cette période.")
        return
    st.dataframe(pd.DataFrame({
        "Date": articles["published_at"].dt.strftime("%d/%m/%Y %H:%M"),
        "Action": articles["ticker"].map(display_name),
        "Source": articles["source"], "Titre": articles["title"],
    }), hide_index=True, width="stretch", height=320)


# -- 2. AI analysis -----------------------------------------------------------------------

def _analysis_options(dataset: dict) -> tuple[AIModel, bool, list] | None:
    """Model and anonymisation choices, shared by the analysis and correlation tabs."""
    models, selected = ai_models()
    ids = [m.id for m in models]
    col1, col2 = st.columns([2, 1])
    model_id = col1.selectbox("Modèle d'IA", ids, index=ids.index(selected),
                              format_func=lambda i: next(m.label for m in models if m.id == i),
                              key="news_model")
    model = next(m for m in models if m.id == model_id)
    col1.caption(model.usage)
    anonymized = col2.checkbox(
        "Anonymiser les sociétés", value=ai_anonymize_default(), key="news_anonymize",
        help="Remplace le nom, les marques et le ticker par « la société A » avant l'envoi à "
             "l'IA, pour qu'elle ne puisse pas s'appuyer sur ce qu'elle sait de la suite.")
    news = st.session_state.get("news")
    if not news or news["articles"].empty:
        st.info("Récupérez d'abord des articles dans l'onglet « 1. Articles ».")
        return None
    jobs = build_jobs(news["articles"], model.id, anonymized)

    if model.knowledge_cutoff is None:
        st.warning("La date de fin des connaissances de ce modèle n'est pas renseignée "
                   "(config/ai_models.yaml). Il peut connaître la suite des événements analysés : "
                   "les résultats sont à lire avec prudence.")
    elif dataset["start"] <= model.knowledge_cutoff:
        st.warning(
            f"⚠️ Une partie de la période analysée est antérieure à la fin des connaissances du "
            f"modèle ({fmt_date(model.knowledge_cutoff)}). Le modèle peut « connaître » ce qu'a "
            "fait l'action après ces articles, ce qui gonfle artificiellement les résultats. "
            "Les indicateurs sont présentés séparément avant et après cette date.")
    return model, anonymized, jobs


def _analysis_tab(dataset: dict) -> None:
    options = _analysis_options(dataset)
    if options is None:
        return
    model, anonymized, jobs = options
    cache = _analysis_cache()

    col1, col2, col3 = st.columns(3)
    limit = col1.number_input("Nombre maximum d'articles à analyser", 0, 100000, 0, step=50,
                              help="0 = tous. Pratique pour un premier essai à petit prix : "
                                   "les articles sont tirés au hasard.")
    batch = col2.checkbox("Traitement différé (moitié prix)", value=False,
                          disabled=not model.supports_batch,
                          help="API Message Batches : résultat en général en moins d'une heure, "
                               "l'écran reste en attente pendant ce temps.")
    max_cost = col3.number_input("Plafond de dépense pour cette analyse ($)", 0.0, 1000.0,
                                 ai_max_cost(), step=0.5, format="%.2f")
    if limit and limit < len(jobs):
        jobs = list(pd.Series(jobs).sample(int(limit), random_state=0))

    estimate = estimate_cost(jobs, cache, model, batch)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles sélectionnés", estimate.n_articles)
    c2.metric("Déjà analysés (gratuits)", estimate.n_cached)
    c3.metric("À envoyer à l'IA", estimate.n_to_analyze)
    c4.metric("Coût estimé", fmt_usd(estimate.cost, 4 if estimate.cost < 0.1 else 2),
              help=f"Estimation : {fmt_number(estimate.input_tokens, 0)} tokens en entrée, "
                   f"{fmt_number(estimate.output_tokens, 0)} en sortie, aux prix de "
                   "config/ai_models.yaml.")
    if estimate.n_to_analyze == 0:
        st.success("Tous les articles sélectionnés sont déjà analysés avec ce modèle.")
        _saved_analyses(cache)
        return
    if model.provider == "anthropic" and not get_api_key(API_KEY_ENV):
        st.warning("La clé API Anthropic n'est pas renseignée (onglet Paramétrage).")
    if estimate.cost > max_cost:
        st.warning(f"Le coût estimé dépasse le plafond ({fmt_usd(max_cost)}) : l'analyse "
                   "s'arrêtera dès que le plafond sera atteint.")
    confirmed = st.checkbox(
        f"Je confirme le lancement de l'analyse de {estimate.n_to_analyze} articles "
        f"(coût estimé {fmt_usd(estimate.cost, 4 if estimate.cost < 0.1 else 2)}, "
        f"plafond {fmt_usd(max_cost)})")
    if st.button("Lancer l'analyse IA", type="primary", disabled=not confirmed):
        bar = st.progress(0.0, text="Analyse en cours…")
        try:
            analyzer = model.make_analyzer()
            report = run_analysis(
                jobs, analyzer, model, cache, max_cost, batch,
                lambda i, n, text: bar.progress(i / max(n, 1), text=text or "Terminé"))
        finally:
            bar.empty()
        st.session_state["news_report"] = report

    report = st.session_state.get("news_report")
    if report:
        if report.fatal_error:
            st.error(f"Analyse interrompue : {report.fatal_error}")
        st.info(f"{report.analyzed} articles analysés, {report.from_cache} repris du cache, "
                f"{len(report.failed)} en échec · coût réel {fmt_usd(report.cost, 4)}")
        if report.stopped_by_cap:
            st.warning("Plafond de dépense atteint : tous les articles n'ont pas été analysés.")
        if report.failed:
            with st.expander(f"Articles en échec ({len(report.failed)})"):
                st.dataframe(pd.DataFrame({"Article": list(report.failed),
                                           "Problème": list(report.failed.values())}),
                             hide_index=True, width="stretch")
    _saved_analyses(cache)


def _saved_analyses(cache: AnalysisCache) -> None:
    """What is stored on disk: every analysis is saved as soon as the AI answers."""
    summary = cache.summary()
    if summary.empty:
        return
    labels = {m.id: m.label for m in ai_models()[0]}
    with st.expander(f"Analyses enregistrées sur ce Mac ({int(summary['n_articles'].sum())} articles, "
                     f"{fmt_usd(summary['cost'].sum())} dépensés)"):
        st.caption("Chaque analyse est enregistrée dès la réponse de l'IA et n'est jamais payée "
                   "deux fois. Elles sont reprises automatiquement dans l'onglet « 3. Corrélations ».")
        st.dataframe(pd.DataFrame({
            "Modèle": summary["model"].map(lambda m: labels.get(m, m)),
            "Anonymisation": summary["anonymized"].map(lambda a: "Oui" if a else "Non"),
            "Articles analysés": summary["n_articles"],
            "Coût": summary["cost"].map(lambda c: fmt_usd(c, 2)),
            "Première analyse": summary["first_at"].map(fmt_date),
            "Dernière analyse": summary["last_at"].map(fmt_date),
        }), hide_index=True, width="stretch")


# -- 3. Correlations ----------------------------------------------------------------------

def _stat_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Format a table of correlation_stats rows; insufficient samples give no conclusion."""
    def corr(v):
        return "—" if v is None or pd.isna(v) else fmt_number(v, 2)

    def pvalue(v):
        return "—" if v is None or pd.isna(v) else ("< 0,001" if v < 0.001 else fmt_number(v, 3))

    rows = {}
    for name, r in table.iterrows():
        if not r["sufficient"]:
            rows[name] = {"Observations": int(r["n"]), "Corrélation de Pearson": "échantillon insuffisant",
                          "p-value (Pearson)": "—", "Corrélation de Spearman": "—",
                          "p-value (Spearman)": "—", "Taux de bon sens": "—"}
        else:
            rows[name] = {"Observations": int(r["n"]), "Corrélation de Pearson": corr(r["pearson"]),
                          "p-value (Pearson)": pvalue(r["pearson_p"]),
                          "Corrélation de Spearman": corr(r["spearman"]),
                          "p-value (Spearman)": pvalue(r["spearman_p"]),
                          "Taux de bon sens": fmt_pct(r["hit_rate"], 1)}
    return pd.DataFrame(rows).T


def _correlation_tab(dataset: dict) -> None:
    options = _analysis_options_readonly()
    if options is None:
        return
    model, jobs, articles = options
    results = load_results(jobs, _analysis_cache())
    if results.empty:
        st.info("Aucun article analysé avec ce modèle et ce réglage d'anonymisation. "
                "Lancez l'analyse dans l'onglet « 2. Analyse IA ».")
        return

    prices, benchmark = dataset["prices"], dataset["benchmark_df"]
    table = article_table(articles, results, prices)
    daily = daily_sentiment(table)
    returns = returns_table(prices, benchmark)
    days = daily.merge(returns, on=["ticker", "day"], how="left")
    per_article = table.merge(returns, on=["ticker", "day"], how="left")
    has_benchmark = benchmark is not None and not benchmark.empty

    col1, col2 = st.columns([2, 1])
    abnormal = col1.radio(
        "Rendements", ["Anormaux (action − indice)", "Bruts"], horizontal=True,
        index=0 if has_benchmark else 1, disabled=not has_benchmark,
        help="Le rendement anormal retire l'effet du marché : rendement de l'action moins "
             "celui de l'indice de référence.").startswith("Anormaux")
    prefix = "ar" if abnormal else "r"
    col2.metric("Jours avec actualités", len(days), f"{table['article_id'].nunique()} articles",
                delta_color="off")
    st.caption(f"Modèle : {model.label} · sentiment du jour = moyenne pondérée par la pertinence "
               f"et la confiance · en dessous de {MIN_OBSERVATIONS} observations, aucune "
               "conclusion n'est affichée. Une corrélation n'est pas une causalité : un article "
               "commente souvent une hausse ou une baisse déjà intervenue (ligne « Jour J »).")

    st.subheader("Sentiment IA et rendements")
    st.dataframe(_stat_frame(horizon_table(days, abnormal=abnormal)), width="stretch")

    before, after = split_by_cutoff(days, model.knowledge_cutoff)
    cutoff_text = (f"le {fmt_date(model.knowledge_cutoff)}" if model.knowledge_cutoff
                   else "une date inconnue (tout est traité comme potentiellement connu du modèle)")
    st.markdown(f"**Avant / après la fin des connaissances du modèle** — {cutoff_text}")
    c1, c2 = st.columns(2)
    c1.caption("Avant : le modèle peut connaître la suite")
    c1.dataframe(_stat_frame(horizon_table(before, abnormal=abnormal))[
        ["Observations", "Corrélation de Pearson", "Taux de bon sens"]], width="stretch")
    c2.caption("Après : le modèle ne peut pas connaître la suite")
    c2.dataframe(_stat_frame(horizon_table(after, abnormal=abnormal))[
        ["Observations", "Corrélation de Pearson", "Taux de bon sens"]], width="stretch")

    strong = strong_days(days, f"{prefix}1")
    c1, c2 = st.columns(2)

    def strong_text(mean, n):
        return "échantillon insuffisant" if mean is None else fmt_pct(mean, signed=True)
    c1.metric("Rendement moyen J+1 après un jour très positif (≥ +0,5)",
              strong_text(strong["mean_positive"], strong["n_positive"]),
              f"{strong['n_positive']} jours", delta_color="off")
    c2.metric("Rendement moyen J+1 après un jour très négatif (≤ −0,5)",
              strong_text(strong["mean_negative"], strong["n_negative"]),
              f"{strong['n_negative']} jours", delta_color="off")

    if days["provider_sentiment"].notna().any():
        st.subheader("Comparaison avec le score de sentiment de la source")
        st.caption("Même calcul avec le score fourni par la source d'actualités : l'analyse IA "
                   "apporte-t-elle quelque chose de plus ?")
        st.dataframe(_stat_frame(horizon_table(days, "provider_sentiment", abnormal)), width="stretch")

    tab_stock, tab_scatter, tab_event, tab_heat, tab_groups, tab_articles = st.tabs(
        ["Cours et sentiment", "Nuage de points", "Étude d'événement", "Par catégorie",
         "Par titre", "Articles analysés"])
    with tab_stock:
        tickers = sorted(daily["ticker"].unique(), key=display_name)
        ticker = st.selectbox("Action", tickers, format_func=display_name, key="news_ticker")
        stock = adjusted_ohlc(prices[ticker]).loc[pd.Timestamp(dataset["start"]):]
        mine = table[table["ticker"] == ticker]
        hover = mine.groupby("day").apply(
            lambda g: "<br>".join(f"• {t[:90]} — {r[:140]}" for t, r in
                                  zip(g["title"].head(6), g["resume_fr"].head(6))),
            include_groups=False)
        st.plotly_chart(charts.sentiment_chart(
            stock, daily[daily["ticker"] == ticker], hover, display_name(ticker)),
            config=charts.PLOTLY_CONFIG)
    with tab_scatter:
        st.plotly_chart(charts.scatter_chart(
            days["sentiment"], days[f"{prefix}1"],
            "Rendement anormal J+1" if abnormal else "Rendement J+1"), config=charts.PLOTLY_CONFIG)
    with tab_event:
        study, counts = event_study(daily, prices, benchmark if abnormal else None)
        if min(counts.values()) < MIN_OBSERVATIONS:
            st.warning(f"Échantillon insuffisant ({counts['positive']} informations positives, "
                       f"{counts['negative']} négatives) : courbes données à titre indicatif.")
        if study.notna().any().any():
            st.plotly_chart(charts.event_chart(study, counts), config=charts.PLOTLY_CONFIG)
            st.caption("Rendement cumulé moyen de J−5 à J+20 autour des informations fortes "
                       "(sentiment ≥ +0,5 ou ≤ −0,5).")
    with tab_heat:
        corr, n_obs = category_heatmap(per_article, abnormal)
        if corr.empty:
            st.info("Aucune donnée.")
        else:
            st.plotly_chart(charts.heatmap_chart(corr, n_obs, CATEGORIES), config=charts.PLOTLY_CONFIG)
            st.caption("Corrélation de Pearson entre le sentiment de chaque article et le "
                       "rendement, par catégorie d'information et par horizon.")
    with tab_groups:
        by_ticker = group_table(days, "ticker", f"{prefix}1")
        by_ticker.index = by_ticker.index.map(display_name)
        st.caption("Sentiment du jour et rendement J+1, par titre")
        st.dataframe(_stat_frame(by_ticker), width="stretch")
    with tab_articles:
        export = pd.DataFrame({
            "Date": per_article["published_at"].dt.strftime("%d/%m/%Y %H:%M"),
            "Jour de bourse": per_article["day"],
            "Action": per_article["ticker"].map(display_name),
            "Source": per_article["source"], "Titre": per_article["title"],
            "Catégorie": per_article["categorie"].map(CATEGORIES),
            "Sentiment": per_article["sentiment"], "Pertinence": per_article["pertinence"],
            "Résumé": per_article["resume_fr"],
            "Rendement J+1 (%)": per_article["r1"] * 100,
            "Rendement J+5 (%)": per_article["r5"] * 100,
        })
        category = st.multiselect("Catégorie", sorted(export["Catégorie"].dropna().unique()))
        view = export[export["Catégorie"].isin(category)] if category else export
        st.dataframe(view.style.format({
            "Jour de bourse": fmt_date, "Sentiment": lambda v: fmt_number(v, 2),
            "Pertinence": lambda v: fmt_number(v, 2),
            "Rendement J+1 (%)": lambda v: fmt_number(v, 2),
            "Rendement J+5 (%)": lambda v: fmt_number(v, 2)}),
            hide_index=True, width="stretch", height=420)
        st.download_button("Exporter en CSV", to_french_csv(view, "articles analysés"),
                           file_name="muge-trading-articles.csv", mime="text/csv")


def _analysis_options_readonly():
    """Same model / anonymisation as the analysis tab, without repeating the widgets."""
    news = st.session_state.get("news")
    if not news or news["articles"].empty:
        st.info("Récupérez d'abord des articles dans l'onglet « 1. Articles ».")
        return None
    models, selected = ai_models()
    model_id = st.session_state.get("news_model", selected)
    model = next((m for m in models if m.id == model_id), models[0])
    anonymized = st.session_state.get("news_anonymize", ai_anonymize_default())
    return model, build_jobs(news["articles"], model.id, anonymized), news["articles"]


def render() -> None:
    st.title("Actualités & IA")
    st.caption("Les informations publiées sur une société sont-elles liées aux mouvements de son "
               "action ? Outil d'analyse : aucun signal de trading n'en est tiré pour l'instant.")
    dataset = st.session_state.get("dataset")
    if not dataset or not dataset.get("tickers"):
        st.info("Commencez par télécharger des cours dans l'écran Données : les actualités "
                "portent sur les mêmes titres et la même période.")
        page_link("data", "Aller à l'écran Données →")
        return
    st.caption(f"{dataset['universe']} · {len(dataset['tickers'])} titres · du "
               f"{fmt_date(dataset['start'])} au {fmt_date(dataset['end'])}")
    restore_news(current_news_provider(), dataset)
    tab_articles, tab_ai, tab_corr = st.tabs(["1. Articles", "2. Analyse IA", "3. Corrélations"])
    with tab_articles:
        _articles_tab(dataset)
    with tab_ai:
        _analysis_tab(dataset)
    with tab_corr:
        _correlation_tab(dataset)
