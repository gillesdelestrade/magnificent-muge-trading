"""AI model catalogue (config/ai_models.yaml): identifiers, prices, knowledge cutoff."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from ..paths import AI_MODELS_PATH
from .analyzer import SentimentAnalyzer


@dataclass(frozen=True)
class AIModel:
    id: str
    label: str
    provider: str = "anthropic"          # anthropic | ollama
    usage: str = ""
    input_price: float = 0.0             # dollars per million tokens
    output_price: float = 0.0
    batch_discount: float = 1.0
    knowledge_cutoff: date | None = None
    base_url: str = ""

    def cost(self, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
        value = (input_tokens * self.input_price + output_tokens * self.output_price) / 1e6
        return value * (self.batch_discount if batch else 1.0)

    @property
    def supports_batch(self) -> bool:
        return self.provider == "anthropic"

    def make_analyzer(self) -> SentimentAnalyzer:
        if self.provider == "ollama":
            from .local_analyzer import OllamaAnalyzer
            return OllamaAnalyzer(self.id, self.base_url or "http://localhost:11434")
        from .claude_analyzer import ClaudeAnalyzer
        return ClaudeAnalyzer(self.id)


def load_models(path: Path | None = None) -> tuple[list[AIModel], str]:
    """Returns (models, id of the default model)."""
    path = path or AI_MODELS_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    models = []
    for m in (raw or {}).get("models", []):
        cutoff = m.get("knowledge_cutoff")
        if isinstance(cutoff, str):
            cutoff = date.fromisoformat(cutoff)
        models.append(AIModel(
            id=str(m["id"]), label=str(m.get("label") or m["id"]),
            provider=str(m.get("provider") or "anthropic"), usage=str(m.get("usage") or ""),
            input_price=float(m.get("input_price") or 0), output_price=float(m.get("output_price") or 0),
            batch_discount=float(m.get("batch_discount") or 1), knowledge_cutoff=cutoff,
            base_url=str(m.get("base_url") or "")))
    if not models:
        models = [AIModel(id="claude-haiku-5-5", label="Claude Haiku 5.5")]
    default = str((raw or {}).get("default") or models[0].id)
    if default not in {m.id for m in models}:
        default = models[0].id
    return models, default


def get_model(model_id: str | None, models: list[AIModel], default: str) -> AIModel:
    by_id = {m.id: m for m in models}
    return by_id.get(model_id or default, by_id[default])
