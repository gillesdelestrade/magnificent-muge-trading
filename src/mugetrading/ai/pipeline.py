"""Analysis pipeline: prompt building, anonymisation, cache, cost estimate, spending cap."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.universes import display_name
from ..paths import AI_CACHE_PATH
from .analyzer import AnalysisError, AnalyzerResult, FatalAnalysisError, SentimentAnalyzer
from .anonymize import anonymize
from .models import AIModel
from .prompts.v1 import PROMPT_VERSION, SYSTEM, build_user_prompt

RESULT_COLUMNS = ["article_id", "ticker", "sentiment", "pertinence", "categorie", "surprise",
                  "horizon", "confiance", "resume_fr", "model", "prompt_version", "anonymized"]
# Rough sizes for the cost estimate (French text: about 3 characters per token).
CHARS_PER_TOKEN = 3.0
OUTPUT_TOKENS_BASE = 250
OUTPUT_TOKENS_PER_TICKER = 150


@dataclass(frozen=True)
class AnalysisJob:
    key: str                       # fingerprint: article + model + prompt version
    article_id: str
    tickers: tuple[str, ...]
    user_prompt: str
    mapping: dict[str, str]        # identifier used in the prompt -> ticker
    anonymized: bool


def build_jobs(articles: pd.DataFrame, model_id: str, anonymized: bool = True,
               names: dict[str, list[str]] | None = None) -> list[AnalysisJob]:
    """One job per article (an article may concern several tickers).

    The prompt contains the article text only: never a price nor a performance.
    """
    jobs = []
    for article_id, group in articles.groupby("id", sort=False):
        first = group.iloc[0]
        tickers = tuple(dict.fromkeys(group["ticker"]))
        text = f"{first['title']}\n\n{first['text']}".strip()
        if anonymized:
            text, mapping = anonymize(text, list(tickers), names)
            companies = [(tag, f"la société {tag}") for tag in mapping]
        else:
            mapping = {t: t for t in tickers}
            companies = [(t, display_name(t)) for t in tickers]
        user_prompt = build_user_prompt(text, companies)
        fingerprint = "\x1f".join([model_id, PROMPT_VERSION, SYSTEM, user_prompt])
        jobs.append(AnalysisJob(
            key=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
            article_id=str(article_id), tickers=tickers, user_prompt=user_prompt,
            mapping=mapping, anonymized=anonymized))
    return jobs


class AnalysisCache:
    """SQLite cache: an article already analysed is never sent to the API again."""

    def __init__(self, path: Path | None = None):
        self.path = path or AI_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS analyses (key TEXT PRIMARY KEY, article_id TEXT, "
                "model TEXT, prompt_version TEXT, anonymized INTEGER, result TEXT, "
                "input_tokens INTEGER, output_tokens INTEGER, cost REAL, created_at TEXT)")

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with closing(sqlite3.connect(self.path)) as con:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for row in con.execute(
                        "SELECT key, article_id, model, prompt_version, anonymized, result "
                        f"FROM analyses WHERE key IN ({marks})", chunk):
                    out[row[0]] = {"article_id": row[1], "model": row[2], "prompt_version": row[3],
                                   "anonymized": bool(row[4]), "analyses": json.loads(row[5])}
        return out

    def get_by_articles(self, article_ids, model_id: str, anonymized: bool) -> pd.DataFrame:
        """Stored analyses of these articles for one model, one row per (article, ticker).

        Looked up by article identifier (much faster than rebuilding every prompt); if an
        article was analysed several times, the most recent analysis is kept.
        """
        wanted = set(map(str, article_ids))
        rows = []
        with closing(sqlite3.connect(self.path)) as con:
            for article_id, result in con.execute(
                    "SELECT article_id, result FROM analyses WHERE model = ? AND anonymized = ? "
                    "AND prompt_version = ? ORDER BY created_at", (model_id, int(anonymized),
                                                                  PROMPT_VERSION)):
                if article_id in wanted:
                    for item in json.loads(result):
                        rows.append({**item, "article_id": article_id, "model": model_id,
                                     "prompt_version": PROMPT_VERSION, "anonymized": anonymized})
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        return frame.drop_duplicates(["article_id", "ticker"], keep="last").reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """What the cache holds, per model: number of articles, cost, first and last analysis."""
        with closing(sqlite3.connect(self.path)) as con:
            return pd.read_sql_query(
                "SELECT model, anonymized, COUNT(*) AS n_articles, COALESCE(SUM(cost), 0) AS cost, "
                "MIN(created_at) AS first_at, MAX(created_at) AS last_at FROM analyses "
                "GROUP BY model, anonymized ORDER BY last_at DESC", con)

    def put(self, job: AnalysisJob, model_id: str, analyses: list[dict],
            input_tokens: int, output_tokens: int, cost: float) -> None:
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute(
                "INSERT OR REPLACE INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.key, job.article_id, model_id, PROMPT_VERSION, int(job.anonymized),
                 json.dumps(analyses, ensure_ascii=False), input_tokens, output_tokens, cost,
                 datetime.now().isoformat(timespec="seconds")))


@dataclass
class CostEstimate:
    n_articles: int          # articles in the selection
    n_cached: int            # already analysed: free
    n_to_analyze: int
    input_tokens: int
    output_tokens: int
    cost: float              # dollars


def estimate_cost(jobs: list[AnalysisJob], cache: AnalysisCache, model: AIModel,
                  batch: bool = False) -> CostEstimate:
    cached = cache.get_many([j.key for j in jobs])
    todo = [j for j in jobs if j.key not in cached]
    input_tokens = sum(_estimate_input(j) for j in todo)
    output_tokens = sum(_estimate_output(j) for j in todo)
    return CostEstimate(len(jobs), len(cached), len(todo), input_tokens, output_tokens,
                        model.cost(input_tokens, output_tokens, batch))


def _estimate_input(job: AnalysisJob) -> int:
    return int((len(SYSTEM) + len(job.user_prompt)) / CHARS_PER_TOKEN) + 300   # + output schema


def _estimate_output(job: AnalysisJob) -> int:
    return OUTPUT_TOKENS_BASE + OUTPUT_TOKENS_PER_TICKER * len(job.tickers)


@dataclass
class RunReport:
    analyzed: int = 0
    from_cache: int = 0
    failed: dict[str, str] = field(default_factory=dict)    # article id -> error
    cost: float = 0.0
    stopped_by_cap: bool = False
    fatal_error: str = ""


def _store(job: AnalysisJob, result: AnalyzerResult, model: AIModel, cache: AnalysisCache,
           batch: bool) -> float:
    analyses = []
    for item in result.analyses:
        ticker = job.mapping.get(item.ticker.strip())
        if ticker is None:      # company outside the list: ignored
            continue
        analyses.append({**item.model_dump(), "ticker": ticker})
    cost = model.cost(result.input_tokens, result.output_tokens, batch)
    cache.put(job, model.id, analyses, result.input_tokens, result.output_tokens, cost)
    return cost


def run_analysis(jobs: list[AnalysisJob], analyzer: SentimentAnalyzer, model: AIModel,
                 cache: AnalysisCache, max_cost: float, batch: bool = False,
                 progress: Callable[[int, int, str], None] | None = None) -> RunReport:
    """Analyse the jobs that are not in the cache, never spending more than ``max_cost`` dollars."""
    report = RunReport()
    cached = cache.get_many([j.key for j in jobs])
    todo = [j for j in jobs if j.key not in cached]
    report.from_cache = len(jobs) - len(todo)
    if not todo:
        return report

    if batch and analyzer.supports_batch:
        # A batch cannot be stopped halfway: only the jobs fitting in the cap are sent.
        selected, planned = [], 0.0
        for job in todo:
            cost = model.cost(_estimate_input(job), _estimate_output(job), batch=True)
            if planned + cost > max_cost:
                report.stopped_by_cap = True
                break
            planned += cost
            selected.append(job)
        if not selected:
            return report
        try:
            results = analyzer.analyze_batch(
                SYSTEM, {j.key: j.user_prompt for j in selected},
                on_wait=(lambda text: progress(0, len(selected), text)) if progress else None)
        except FatalAnalysisError as exc:
            report.fatal_error = str(exc)
            return report
        for job in selected:
            result = results.get(job.key, AnalysisError("Résultat absent du lot"))
            if isinstance(result, Exception):
                report.failed[job.article_id] = str(result)
            else:
                report.cost += _store(job, result, model, cache, batch=True)
                report.analyzed += 1
        return report

    for i, job in enumerate(todo):
        if progress:
            progress(i, len(todo), f"Article {i + 1}/{len(todo)}")
        next_cost = model.cost(_estimate_input(job), _estimate_output(job))
        if report.cost + next_cost > max_cost:
            report.stopped_by_cap = True
            break
        try:
            result = analyzer.analyze(SYSTEM, job.user_prompt)
        except FatalAnalysisError as exc:
            report.fatal_error = str(exc)
            break
        except AnalysisError as exc:
            report.failed[job.article_id] = str(exc)
            continue
        report.cost += _store(job, result, model, cache, batch=False)
        report.analyzed += 1
    if progress:
        progress(len(todo), len(todo), "")
    return report


def load_results(jobs: list[AnalysisJob], cache: AnalysisCache) -> pd.DataFrame:
    """Cached analyses of the jobs, one row per (article, ticker)."""
    cached = cache.get_many([j.key for j in jobs])
    rows = []
    for job in jobs:
        entry = cached.get(job.key)
        if not entry:
            continue
        for item in entry["analyses"]:
            rows.append({**item, "article_id": job.article_id, "model": entry["model"],
                         "prompt_version": entry["prompt_version"],
                         "anonymized": entry["anonymized"]})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
