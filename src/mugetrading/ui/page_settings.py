"""Settings tab: data source, default values, fee profiles."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from ..data.registry import list_providers
from ..engine.fees import FeeBracket, FeeProfile, save_profiles
from ..env import get_api_key, set_api_key
from ..ai.pipeline import AnalysisCache
from ..formatting import fmt_date, fmt_eur, fmt_number, fmt_usd
from ..paths import BACKUP_DIR
from ..storage.backup import create_backup, list_backups
from ..strategy.params import MIN_CAPITAL
from ..news.provider import list_news_providers
from .common import (SETTING_AI_ANONYMIZE, SETTING_AI_MAX_COST, SETTING_AI_MODEL, SETTING_CAPITAL,
                     SETTING_NEWS_SOURCE, SETTING_PROFILE, SETTING_SOURCE, ai_anonymize_default,
                     ai_max_cost, ai_models, current_news_provider, current_provider,
                     default_capital, default_profile_name, fee_profiles, get_db)

TEST_TICKER = "MC.PA"
NEW_PROFILE = "➕ Nouveau profil"
BRACKET_COLUMNS = ["jusqu_a", "fixe", "pourcentage", "minimum", "maximum"]


def _source_section() -> None:
    st.subheader("Source de données de marché")
    db = get_db()
    providers = list_providers()
    active = current_provider()

    def label(p) -> str:
        return p.label if p.available else f"{p.label} — Bientôt disponible"

    chosen = st.selectbox(
        "Source", providers, index=providers.index(active), format_func=label,
        help="Les sources marquées « Bientôt disponible » ne sont pas encore utilisables.")
    st.caption(chosen.description)
    if not chosen.available:
        st.info(f"{chosen.label} n'est pas encore disponible. "
                f"La source utilisée reste {active.label}.")
    elif chosen.id != active.id:
        db.set_setting(SETTING_SOURCE, chosen.id)
        st.success(f"Source enregistrée : {chosen.label}")

    if chosen.needs_api_key and chosen.api_key_env:
        key = st.text_input("Clé API", value=get_api_key(chosen.api_key_env), type="password",
                            disabled=not chosen.available,
                            help="Conservée dans le fichier .env local, jamais en base ni dans git.")
        if chosen.available and st.button("Enregistrer la clé"):
            set_api_key(chosen.api_key_env, key)
            st.success("Clé enregistrée dans le fichier .env")

    if st.button("Tester la connexion", disabled=not chosen.available):
        end = date.today()
        try:
            with st.spinner(f"Téléchargement de quelques jours de {TEST_TICKER}…"):
                df = chosen.fetch_daily(TEST_TICKER, end - timedelta(days=10), end)
        except Exception as exc:
            st.error(f"Échec de la connexion à {chosen.label} : {exc}")
        else:
            if df.empty:
                st.error(f"{chosen.label} a répondu mais n'a renvoyé aucun cours pour {TEST_TICKER}.")
            else:
                last = df.iloc[-1]
                st.success(
                    f"Connexion réussie : {len(df)} jours reçus pour {TEST_TICKER}, dernière "
                    f"clôture {fmt_eur(last['close'])} le {fmt_date(df.index[-1])}.")


def _api_key_field(label: str, env_name: str, key: str) -> None:
    """Masked input; the stored key is never displayed, only replaced."""
    stored = bool(get_api_key(env_name))
    value = st.text_input(
        label, type="password", key=key,
        placeholder="Clé déjà enregistrée — saisir pour la remplacer" if stored else "Coller la clé ici",
        help="Conservée dans le fichier .env local, jamais en base ni dans git.")
    if st.button("Enregistrer la clé", key=f"{key}_save", disabled=not value.strip()):
        set_api_key(env_name, value.strip())
        st.success("Clé enregistrée dans le fichier .env")


def _news_ai_section() -> None:
    st.subheader("Actualités & IA")
    db = get_db()
    col1, col2 = st.columns(2)
    with col1:
        providers = list_news_providers()
        active = current_news_provider()
        chosen = st.selectbox(
            "Source d'actualités", providers,
            index=next(i for i, p in enumerate(providers) if p.id == active.id),
            format_func=lambda p: p.label if p.available else f"{p.label} — Bientôt disponible")
        st.caption(chosen.description)
        if not chosen.available:
            st.info(f"{chosen.label} n'est pas encore disponible. "
                    f"La source utilisée reste {active.label}.")
        elif chosen.id != active.id:
            db.set_setting(SETTING_NEWS_SOURCE, chosen.id)
            st.success(f"Source d'actualités enregistrée : {chosen.label}")
        if chosen.available and chosen.needs_api_key and chosen.api_key_env:
            _api_key_field(f"Clé API {chosen.label}", chosen.api_key_env, "news_key")
    with col2:
        models, selected = ai_models()
        ids = [m.id for m in models]
        model_id = st.selectbox("Modèle d'IA", ids, index=ids.index(selected),
                                format_func=lambda i: next(m.label for m in models if m.id == i))
        model = next(m for m in models if m.id == model_id)
        cutoff = fmt_date(model.knowledge_cutoff) if model.knowledge_cutoff else "non renseignée"
        price = ("gratuit" if not model.input_price and not model.output_price else
                 f"{model.input_price:g} $ / {model.output_price:g} $ par million de tokens "
                 "(entrée / sortie)")
        st.caption(f"{model.usage} · {price} · fin des connaissances : {cutoff}. "
                   "Prix et dates modifiables dans config/ai_models.yaml.")
        if model.provider == "anthropic":
            _api_key_field("Clé API Anthropic", "ANTHROPIC_API_KEY", "anthropic_key")
        else:
            st.caption(f"Modèle local servi par Ollama ({model.base_url or 'http://localhost:11434'}) : "
                       "aucune clé nécessaire.")
    col1, col2 = st.columns(2)
    cap = col1.number_input("Plafond de dépense par analyse ($)", 0.0, 1000.0, ai_max_cost(),
                            step=0.5, format="%.2f",
                            help="L'analyse s'arrête dès que ce montant est atteint.")
    anonymize = col2.checkbox(
        "Anonymiser les sociétés avant l'envoi à l'IA", value=ai_anonymize_default(),
        help="Remplace le nom, les marques et le ticker par « la société A », pour que le modèle "
             "ne s'appuie pas sur ce qu'il sait de la suite des événements.")
    if st.button("Enregistrer les réglages Actualités & IA"):
        db.set_setting(SETTING_AI_MODEL, model_id)
        db.set_setting(SETTING_AI_MAX_COST, float(cap))
        db.set_setting(SETTING_AI_ANONYMIZE, bool(anonymize))
        for key in ("news_model", "news_anonymize"):
            st.session_state.pop(key, None)
        st.success("Réglages enregistrés.")


def _defaults_section(profiles: list[FeeProfile]) -> None:
    st.subheader("Valeurs par défaut des nouvelles simulations")
    db = get_db()
    col1, col2 = st.columns(2)
    capital = col1.number_input(
        "Capital de départ par défaut (€)", min_value=MIN_CAPITAL, value=default_capital(),
        step=100.0, format="%.2f", help="Modifiable ensuite pour chaque simulation.")
    names = [p.name for p in profiles]
    profile = col2.selectbox("Plateforme (profil de frais) par défaut", names,
                             index=names.index(default_profile_name(profiles)))
    if st.button("Enregistrer les valeurs par défaut"):
        db.set_setting(SETTING_CAPITAL, round(float(capital), 2))
        db.set_setting(SETTING_PROFILE, profile)
        st.session_state.pop("sim_capital", None)
        st.session_state.pop("sim_profile", None)
        st.success("Valeurs par défaut enregistrées.")


def _none(value):
    return None if value is None or pd.isna(value) else float(value)


def _profiles_section(profiles: list[FeeProfile]) -> None:
    st.subheader("Profils de frais des plateformes")
    st.caption("Par tranche : frais = min(maximum, max(minimum, fixe + pourcentage × montant)). "
               "Les modifications sont écrites dans config/fee_profiles.yaml.")
    names = [p.name for p in profiles]
    choice = st.selectbox("Profil à modifier", names + [NEW_PROFILE])
    is_new = choice == NEW_PROFILE
    current = FeeProfile(name="", verified_on=date.today()) if is_new else profiles[names.index(choice)]
    if not is_new and current.is_stale():
        st.warning(f"Grille vérifiée le {fmt_date(current.verified_on)} : elle date de plus de "
                   "6 mois, pensez à la revérifier.")

    key = f"profile_{choice}"
    col1, col2 = st.columns([2, 1])
    name = col1.text_input("Nom", value=current.name, key=f"{key}_name")
    account = col2.selectbox("Type de compte", ["CTO", "PEA"],
                             index=0 if current.account_type != "PEA" else 1, key=f"{key}_acc",
                             help="En PEA, les frais sont plafonnés à 0,5 % du montant de l'ordre.")
    description = st.text_input("Description", value=current.description, key=f"{key}_desc")
    rows = pd.DataFrame([{
        "jusqu_a": b.up_to, "fixe": b.fixed, "pourcentage": b.pct,
        "minimum": b.minimum, "maximum": b.maximum} for b in current.brackets],
        columns=BRACKET_COLUMNS).astype("float64")
    edited = st.data_editor(
        rows, num_rows="dynamic", hide_index=True, key=f"{key}_brackets",
        column_config={
            "jusqu_a": st.column_config.NumberColumn(
                "Jusqu'à (€)", min_value=0.0, help="Montant maximum de l'ordre pour cette tranche "
                "(inclus). Laisser vide pour la dernière tranche."),
            "fixe": st.column_config.NumberColumn("Fixe (€)", min_value=0.0, format="%.2f"),
            "pourcentage": st.column_config.NumberColumn(
                "Pourcentage (%)", min_value=0.0, format="%.3f", help="0,09 signifie 0,09 % du montant."),
            "minimum": st.column_config.NumberColumn("Minimum (€)", min_value=0.0, format="%.2f"),
            "maximum": st.column_config.NumberColumn("Maximum (€)", min_value=0.0, format="%.2f"),
        })
    col1, col2, col3 = st.columns(3)
    free_n = col1.number_input("Ordres gratuits par mois", min_value=0, step=1,
                               value=current.free_orders_per_month, key=f"{key}_free")
    free_max = col2.number_input("… jusqu'à un montant de (€)", min_value=0.0, step=50.0,
                                 value=float(current.free_orders_max_amount), key=f"{key}_freemax")
    verified = col3.date_input("Grille vérifiée le", value=current.verified_on or date.today(),
                               format="DD/MM/YYYY", key=f"{key}_ver")
    url = st.text_input("Lien vers la grille officielle", value=current.source_url, key=f"{key}_url")
    if current.source_url:
        st.caption(f"[Ouvrir la grille officielle]({current.source_url})")

    col_save, col_delete = st.columns([1, 1])
    if col_save.button("Enregistrer le profil", type="primary"):
        brackets = tuple(
            FeeBracket(up_to=_none(r.jusqu_a), fixed=_none(r.fixe) or 0.0, pct=_none(r.pourcentage) or 0.0,
                       minimum=_none(r.minimum), maximum=_none(r.maximum))
            for r in edited.itertuples())
        clean = name.strip()
        if not clean:
            st.error("Le profil doit avoir un nom.")
        elif not brackets:
            st.error("Le profil doit contenir au moins une tranche.")
        elif clean != current.name and clean in names:
            st.error("Un autre profil porte déjà ce nom.")
        else:
            profile = FeeProfile(
                name=clean, description=description.strip(), brackets=brackets,
                free_orders_per_month=int(free_n), free_orders_max_amount=float(free_max),
                account_type=account, verified_on=verified, source_url=url.strip())
            updated = profiles + [profile] if is_new else [
                profile if p.name == current.name else p for p in profiles]
            save_profiles(updated)
            st.success(f"Profil « {clean} » enregistré.")
    if not is_new and len(profiles) > 1:
        confirm = col_delete.checkbox("Confirmer la suppression", key=f"{key}_confirm")
        if col_delete.button("Supprimer le profil", disabled=not confirm):
            save_profiles([p for p in profiles if p.name != current.name])
            st.rerun()


def _backup_section() -> None:
    st.subheader("Sauvegarde des données")
    db = get_db()
    listing = db.list_simulations()
    try:
        analyses = AnalysisCache().summary()
    except Exception:
        analyses = pd.DataFrame(columns=["n_articles", "cost"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulations enregistrées", len(listing),
              f"dont {int((~listing['auto'].astype(bool)).sum())} nommées", delta_color="off")
    c2.metric("Articles analysés par l'IA", int(analyses["n_articles"].sum()))
    c3.metric("Dépense IA enregistrée", fmt_usd(analyses["cost"].sum()))
    st.caption(
        "Tout est enregistré automatiquement sur ce Mac, dans le dossier `data/` de "
        "l'application : simulations, réglages, articles et analyses de l'IA. Au prochain "
        "lancement, vous retrouvez vos données, votre dernière simulation et vos articles. "
        "Une sauvegarde est une copie de sécurité dans un fichier zip, à ranger ailleurs "
        "(clé USB, autre ordinateur). Elle ne contient ni les cours (retéléchargeables "
        "gratuitement) ni les clés API.")
    if st.button("Créer une sauvegarde"):
        with st.spinner("Création de la sauvegarde…"):
            path = create_backup()
        st.success(f"Sauvegarde créée : {path.name}")
    backups = list_backups()
    if backups:
        latest = backups[0]
        st.caption(f"{len(backups)} sauvegarde(s) dans le dossier `{BACKUP_DIR.name}/` · dernière : "
                   f"{latest.name} ({fmt_number(latest.stat().st_size / 1e6, 1)} Mo)")
        with latest.open("rb") as handle:
            st.download_button("Télécharger la dernière sauvegarde", handle.read(),
                               file_name=latest.name, mime="application/zip")
        st.caption("Pour restaurer : fermer l'application, puis décompresser le fichier zip "
                   "dans le dossier de l'application en acceptant de remplacer les fichiers.")


def render() -> None:
    st.title("Paramétrage")
    profiles = fee_profiles()
    _source_section()
    st.divider()
    _news_ai_section()
    st.divider()
    _defaults_section(profiles)
    st.divider()
    _profiles_section(profiles)
    st.divider()
    _backup_section()
