"""Structured output of the AI analysis: one entry per company concerned by the article."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

CATEGORIES = {
    "resultats": "Résultats",
    "previsions": "Prévisions",
    "fusion_acquisition": "Fusion-acquisition",
    "dividende_rachat": "Dividende / rachat d'actions",
    "contrat_commande": "Contrat / commande",
    "dirigeants": "Dirigeants",
    "juridique_reglementaire": "Juridique / réglementaire",
    "note_analyste": "Note d'analyste",
    "macro_secteur": "Macroéconomie / secteur",
    "autre": "Autre",
}
HORIZONS = {"immediat": "Immédiat", "court_terme": "Court terme", "long_terme": "Long terme"}

Categorie = Literal["resultats", "previsions", "fusion_acquisition", "dividende_rachat",
                    "contrat_commande", "dirigeants", "juridique_reglementaire",
                    "note_analyste", "macro_secteur", "autre"]
Horizon = Literal["immediat", "court_terme", "long_terme"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class TickerAnalysis(BaseModel):
    ticker: str            # identifier of the company as given in the prompt
    sentiment: float       # -1 (very negative for the share) .. +1 (very positive)
    pertinence: float      # 0..1: is the article really about this company?
    categorie: Categorie
    surprise: float        # 0 (expected) .. 1 (unexpected)
    horizon: Horizon
    confiance: float       # 0..1: confidence of the model in its own reading
    resume_fr: str         # one-sentence summary, in French

    @field_validator("sentiment")
    @classmethod
    def _sentiment_range(cls, v: float) -> float:
        return _clamp(v, -1.0, 1.0)

    @field_validator("pertinence", "surprise", "confiance")
    @classmethod
    def _unit_range(cls, v: float) -> float:
        return _clamp(v, 0.0, 1.0)


class ArticleAnalysis(BaseModel):
    analyses: list[TickerAnalysis]


# JSON schema sent to the model (structured outputs). Numeric bounds are not expressed
# here (not supported by constrained decoding): they are described in the prompt and
# enforced by the pydantic validators above.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "sentiment": {"type": "number"},
                    "pertinence": {"type": "number"},
                    "categorie": {"type": "string", "enum": list(CATEGORIES)},
                    "surprise": {"type": "number"},
                    "horizon": {"type": "string", "enum": list(HORIZONS)},
                    "confiance": {"type": "number"},
                    "resume_fr": {"type": "string"},
                },
                "required": ["ticker", "sentiment", "pertinence", "categorie", "surprise",
                             "horizon", "confiance", "resume_fr"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["analyses"],
    "additionalProperties": False,
}
