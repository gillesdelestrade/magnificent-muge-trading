"""Claude models (Haiku / Sonnet / Fable) through the official anthropic SDK."""
from __future__ import annotations

import time
from typing import Callable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ..env import get_api_key
from .analyzer import (AnalysisError, AnalyzerResult, FatalAnalysisError, SentimentAnalyzer,
                       parse_output)
from .schema import OUTPUT_SCHEMA

API_KEY_ENV = "ANTHROPIC_API_KEY"
MAX_TOKENS = 4000          # short structured answer; leaves room for the model's thinking
BATCH_POLL_SECONDS = 30


def make_client() -> anthropic.Anthropic:
    key = get_api_key(API_KEY_ENV)
    if not key:
        raise FatalAnalysisError("Clé API Anthropic manquante (onglet Paramétrage).")
    return anthropic.Anthropic(api_key=key)


class ClaudeAnalyzer(SentimentAnalyzer):
    supports_batch = True

    def __init__(self, model_id: str, client=None, sleep: Callable[[float], None] = time.sleep):
        self.model_id = model_id
        self._client = client
        self._sleep = sleep

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def _params(self, system: str, user: str) -> dict:
        return dict(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Simple reading task: low effort keeps it fast and cheap.
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )

    @staticmethod
    def _to_result(message) -> AnalyzerResult:
        if message.stop_reason == "refusal":
            raise AnalysisError("Le modèle a refusé d'analyser cet article.")
        if message.stop_reason == "max_tokens":
            raise AnalysisError("Réponse tronquée (limite de tokens atteinte).")
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            raise AnalysisError("Réponse vide du modèle.")
        return AnalyzerResult(parse_output(text), message.usage.input_tokens,
                              message.usage.output_tokens)

    def analyze(self, system: str, user: str) -> AnalyzerResult:
        try:
            message = self.client.messages.create(**self._params(system, user))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise FatalAnalysisError("Clé API Anthropic refusée.") from exc
        except anthropic.NotFoundError as exc:
            raise FatalAnalysisError(f"Modèle inconnu : {self.model_id}") from exc
        except anthropic.BadRequestError as exc:
            raise AnalysisError(f"Requête refusée par l'API : {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise FatalAnalysisError("Limite de débit de l'API atteinte : réessayez plus tard.") from exc
        except anthropic.APIStatusError as exc:
            raise AnalysisError(f"Erreur de l'API ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise FatalAnalysisError("Connexion à l'API Anthropic impossible.") from exc
        return self._to_result(message)

    def analyze_batch(self, system: str, prompts: dict[str, str],
                      on_wait: Callable[[str], None] | None = None,
                      ) -> dict[str, AnalyzerResult | AnalysisError]:
        """Message Batches API: deferred processing, half price. Blocks until the batch ends."""
        try:
            batch = self.client.messages.batches.create(requests=[
                Request(custom_id=key,
                        params=MessageCreateParamsNonStreaming(**self._params(system, user)))
                for key, user in prompts.items()])
            while batch.processing_status != "ended":
                if on_wait:
                    counts = batch.request_counts
                    on_wait(f"Lot {batch.id} en cours : {counts.succeeded + counts.errored} "
                            f"terminés, {counts.processing} en attente")
                self._sleep(BATCH_POLL_SECONDS)
                batch = self.client.messages.batches.retrieve(batch.id)
            results: dict[str, AnalyzerResult | AnalysisError] = {}
            for entry in self.client.messages.batches.results(batch.id):
                if entry.result.type == "succeeded":
                    try:
                        results[entry.custom_id] = self._to_result(entry.result.message)
                    except AnalysisError as exc:
                        results[entry.custom_id] = exc
                else:
                    results[entry.custom_id] = AnalysisError(f"Requête du lot : {entry.result.type}")
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise FatalAnalysisError("Clé API Anthropic refusée.") from exc
        except anthropic.APIStatusError as exc:
            raise FatalAnalysisError(f"Erreur de l'API ({exc.status_code}) : {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise FatalAnalysisError("Connexion à l'API Anthropic impossible.") from exc
        return results
