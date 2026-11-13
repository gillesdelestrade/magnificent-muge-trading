"""Magnificent Muge Trading — Streamlit entry point (streamlit run app.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from mugetrading import APP_NAME  # noqa: E402
from mugetrading.paths import FAVICON_PATH, LOGO_PATH  # noqa: E402
from mugetrading.ui.session import restore_session  # noqa: E402
from mugetrading.ui import (page_compare, page_data, page_home, page_news_ai,  # noqa: E402
                            page_results, page_settings, page_simulation)

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🐟",
    layout="wide",
)

# Gold accent for titles and key figures (the rest of the theme is in .streamlit/config.toml)
st.markdown(
    "<style>h1, h2, h3 {color: #D4A85A;} "
    "[data-testid='stMetricValue'] {color: #D4A85A;} "
    "[data-testid='stLogo'] {height: 5.5rem;}</style>",
    unsafe_allow_html=True,
)

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")
else:
    st.sidebar.markdown(f"### {APP_NAME}")

pages = {
    "home": st.Page(page_home.render, title="Accueil", icon=":material/home:", url_path="accueil", default=True),
    "data": st.Page(page_data.render, title="Données", icon=":material/download:", url_path="donnees"),
    "simulation": st.Page(page_simulation.render, title="Simulation", icon=":material/tune:", url_path="simulation"),
    "results": st.Page(page_results.render, title="Résultats", icon=":material/monitoring:", url_path="resultats"),
    "compare": st.Page(page_compare.render, title="Comparaison", icon=":material/compare_arrows:", url_path="comparaison"),
    "news": st.Page(page_news_ai.render, title="Actualités & IA", icon=":material/newspaper:", url_path="actualites"),
    "settings": st.Page(page_settings.render, title="Paramétrage", icon=":material/settings:", url_path="parametrage"),
}
st.session_state["pages"] = pages
with st.spinner("Reprise de la dernière session…"):
    restore_session()
st.navigation(list(pages.values())).run()
