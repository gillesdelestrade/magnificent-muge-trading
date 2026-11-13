"""SentimentAnalyzer interface: every AI model (Claude, local) implements it,
so models can be compared on the same articles."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .schema import ArticleAnalysis, TickerAnalysis


class AnalysisError(Exception):
    """This article could not be analysed (the run continues with the next one)."""


class FatalAnalysisError(AnalysisError):
    """The whole run must stop (bad API key, model not found, server unreachable...)."""


@dataclass
class AnalyzerResult:
    analyses: list[TickerAnalysis]
    input_tokens: int = 0
    output_tokens: int = 0


def parse_output(text: str) -> list[TickerAnalysis]:
    """Validate the JSON returned by a model against the pydantic schema."""
    try:
        return ArticleAnalysis.model_validate_json(text).analyses
    except Exception as exc:
        raise AnalysisError(f"Réponse du modèle non conforme au schéma : {exc}") from exc


class SentimentAnalyzer(ABC):
    model_id: str = ""
    supports_batch: bool = False

    @abstractmethod
    def analyze(self, system: str, user: str) -> AnalyzerResult:
        """Analyse one article (the prompt only contains the article text)."""

    def analyze_batch(self, system: str, prompts: dict[str, str],
                      on_wait: Callable[[str], None] | None = None,
                      ) -> dict[str, AnalyzerResult | AnalysisError]:
        """Deferred processing of many articles (key -> result or error)."""
        raise NotImplementedError
