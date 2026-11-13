"""Local model through Ollama's HTTP API (free, runs on the machine)."""
from __future__ import annotations

from typing import Callable

import requests

from .analyzer import (AnalysisError, AnalyzerResult, FatalAnalysisError, SentimentAnalyzer,
                       parse_output)
from .schema import OUTPUT_SCHEMA


class OllamaAnalyzer(SentimentAnalyzer):
    def __init__(self, model_id: str, base_url: str = "http://localhost:11434",
                 http_post: Callable | None = None):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self._post = http_post or requests.post

    def analyze(self, system: str, user: str) -> AnalyzerResult:
        try:
            response = self._post(f"{self.base_url}/api/chat", json={
                "model": self.model_id, "stream": False, "format": OUTPUT_SCHEMA,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }, timeout=300)
        except Exception as exc:
            raise FatalAnalysisError(
                f"Ollama injoignable sur {self.base_url} : est-il lancé ?") from exc
        if response.status_code == 404:
            raise FatalAnalysisError(
                f"Modèle local « {self.model_id} » introuvable (ollama pull {self.model_id}).")
        if response.status_code != 200:
            raise AnalysisError(f"Ollama : erreur HTTP {response.status_code}")
        data = response.json()
        text = (data.get("message") or {}).get("content") or ""
        return AnalyzerResult(parse_output(text), int(data.get("prompt_eval_count") or 0),
                              int(data.get("eval_count") or 0))
