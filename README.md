# Magnificent Muge Trading

Simulateur de trading (backtest) sur données historiques réelles, avec un volet d'analyse des
actualités par IA. Application personnelle, entièrement locale. Aucun ordre réel n'est passé.

## Installation

Il faut [uv](https://docs.astral.sh/uv/) (macOS Apple Silicon ou Intel) :

```
uv sync
```

## Lancement

```
uv run streamlit run app.py
```

L'application s'ouvre dans le navigateur sur http://localhost:8501.

## Clés API (facultatives)

Yahoo Finance ne demande aucune clé. Pour les actualités EODHD et l'analyse IA (Claude), les clés
se saisissent dans l'onglet Paramétrage ; elles sont rangées dans un fichier `.env` local, jamais
envoyé sur GitHub (modèle : `.env.example`).

## Tests

```
uv run pytest
```

Le cahier des charges est dans `CLAUDE.md` et `CLAUDE avec infos et ia.md`.
