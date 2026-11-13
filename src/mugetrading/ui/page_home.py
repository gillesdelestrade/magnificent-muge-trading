"""Home page."""
from __future__ import annotations

import streamlit as st

from .. import APP_NAME
from ..paths import LOGO_PATH
from .common import current_provider, page_link


def render() -> None:
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=320)
    st.markdown(f"<h1 style='text-align:center;color:#D4A85A'>{APP_NAME}</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center'>Testez des stratégies simples d'achat et de vente d'actions "
        "sur des cours historiques réels, avant de risquer le moindre euro.</p>",
        unsafe_allow_html=True)
    st.divider()
    cols = st.columns(4)
    steps = [
        ("data", "1. Données", "Choisir les actions et la période, télécharger les cours."),
        ("simulation", "2. Simulation", "Régler le capital, les signaux, les stops et les frais."),
        ("results", "3. Résultats", "Journal des opérations, indicateurs et graphiques."),
        ("compare", "4. Comparaison", "Rouvrir et comparer les simulations sauvegardées."),
    ]
    for col, (key, title, text) in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
            page_link(key, "Ouvrir")
    st.caption(
        f"Source de données : {current_provider().label} · Simulation uniquement : "
        "aucun ordre réel n'est envoyé. Les performances passées ne préjugent pas des "
        "performances futures.")
