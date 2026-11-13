"""AI tests: every call to a model is simulated — no paid call during pytest."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mugetrading.ai.analyzer import (AnalysisError, AnalyzerResult, FatalAnalysisError,
                                     SentimentAnalyzer, parse_output)
from mugetrading.ai.anonymize import anonymize, company_names, label
from mugetrading.ai.claude_analyzer import ClaudeAnalyzer
from mugetrading.ai.local_analyzer import OllamaAnalyzer
from mugetrading.ai.models import AIModel, load_models
from mugetrading.ai.pipeline import (AnalysisCache, build_jobs, estimate_cost, load_results,
                                     run_analysis)
from mugetrading.ai.prompts.v1 import SYSTEM
from mugetrading.ai.schema import OUTPUT_SCHEMA, TickerAnalysis
from mugetrading.news.provider import normalize_articles

HAIKU = AIModel(id="claude-haiku-5-5", label="Haiku", input_price=1.0, output_price=5.0,
                batch_discount=0.5)


def entry(ticker="A", sentiment=0.6, **kw):
    base = dict(ticker=ticker, sentiment=sentiment, pertinence=0.9, categorie="resultats",
                surprise=0.5, horizon="court_terme", confiance=0.8, resume_fr="Bons résultats.")
    base.update(kw)
    return base


def articles():
    return normalize_articles(pd.DataFrame([
        {"published_at": "2024-01-05T09:00:00+00:00", "source": "s", "url": "https://x/1",
         "title": "LVMH (MC.PA) bat les attentes", "ticker": "MC.PA",
         "text": "Le groupe de Bernard Arnault profite de Louis Vuitton et de Dior. Kering recule."},
        {"published_at": "2024-01-05T09:00:00+00:00", "source": "s", "url": "https://x/1",
         "title": "LVMH (MC.PA) bat les attentes", "ticker": "KER.PA",
         "text": "Le groupe de Bernard Arnault profite de Louis Vuitton et de Dior. Kering recule."},
        {"published_at": "2024-01-08T09:00:00+00:00", "source": "s", "url": "https://x/2",
         "title": "Airbus livre moins d'avions", "ticker": "AIR.PA", "text": "Retards chez Airbus."},
    ]))


class FakeAnalyzer(SentimentAnalyzer):
    model_id = "claude-haiku-5-5"
    supports_batch = True

    def __init__(self, fail_on=None):
        self.calls, self.batch_calls, self.fail_on = [], [], fail_on

    def _result(self, user):
        if self.fail_on and self.fail_on in user:
            raise AnalysisError("refus")
        tags = [line[2:].split(" : ")[0] for line in user.splitlines() if line.startswith("- ")]
        return AnalyzerResult([TickerAnalysis(**entry(t)) for t in tags] +
                              [TickerAnalysis(**entry("HORS_LISTE"))], 1000, 200)

    def analyze(self, system, user):
        self.calls.append(user)
        return self._result(user)

    def analyze_batch(self, system, prompts, on_wait=None):
        self.batch_calls.append(list(prompts))
        return {k: self._result(u) for k, u in prompts.items()}


# -- anonymisation -----------------------------------------------------------------

def test_anonymize_removes_name_brands_and_ticker():
    text = ("LVMH (MC.PA) et sa marque Louis Vuitton progressent ; MC gagne 2 %. "
            "L'action de Moët Hennessy Louis Vuitton, dirigée par Bernard Arnault, dépasse Kering.")
    out, mapping = anonymize(text, ["MC.PA", "KER.PA"])
    assert mapping == {"A": "MC.PA", "B": "KER.PA"}
    for hidden in ("LVMH", "MC.PA", "Louis Vuitton", "Moët", "Arnault", "Kering"):
        assert hidden not in out
    assert out.count("la société A") == 5 and out.count("la société B") == 1


def test_anonymize_is_case_and_accent_tolerant():
    out, _ = anonymize("HERMES et hermès, l'oreal et L’Oréal", ["RMS.PA", "OR.PA"])
    assert out == "la société A et la société A, la société B et la société B"


def test_anonymize_keeps_common_words_that_look_like_short_tickers():
    # "AI" is Air Liquide's code but also a common word: only replaced in "(AI)"
    out, _ = anonymize("AI boom: Air Liquide (AI) invests in AI", ["AI.PA"])
    assert out == "AI boom: la société A (la société A) invests in AI"


def test_labels_and_names():
    assert [label(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]
    assert "Hermes" in company_names("RMS.PA") and "Hermès" in company_names("RMS.PA")


# -- schema ------------------------------------------------------------------------

def test_schema_validation_and_clamping():
    out = parse_output(json.dumps({"analyses": [entry(sentiment=3.0, pertinence=-1, confiance=1.5)]}))
    assert (out[0].sentiment, out[0].pertinence, out[0].confiance) == (1.0, 0.0, 1.0)
    with pytest.raises(AnalysisError):
        parse_output(json.dumps({"analyses": [entry(categorie="inconnue")]}))
    with pytest.raises(AnalysisError):
        parse_output("pas du json")
    assert set(OUTPUT_SCHEMA["properties"]["analyses"]["items"]["required"]) == set(TickerAnalysis.model_fields)


# -- pipeline ----------------------------------------------------------------------

def test_jobs_one_per_article_prompt_has_only_the_text():
    jobs = build_jobs(articles(), "claude-haiku-5-5", anonymized=True)
    assert len(jobs) == 2 and jobs[0].tickers == ("KER.PA", "MC.PA")
    assert "LVMH" not in jobs[0].user_prompt and "Kering" not in jobs[0].user_prompt
    assert "la société A" in jobs[0].user_prompt and jobs[0].mapping == {"A": "KER.PA", "B": "MC.PA"}
    clear = build_jobs(articles(), "claude-haiku-5-5", anonymized=False)
    assert "LVMH" in clear[0].user_prompt and clear[0].key != jobs[0].key
    # the fingerprint depends on the model
    assert build_jobs(articles(), "claude-sonnet-5-5")[0].key != jobs[0].key


def test_cache_prevents_any_second_call(tmp_path):
    cache = AnalysisCache(tmp_path / "ai.sqlite")
    jobs = build_jobs(articles(), "claude-haiku-5-5")
    analyzer = FakeAnalyzer()
    first = run_analysis(jobs, analyzer, HAIKU, cache, max_cost=10)
    assert (first.analyzed, first.from_cache, len(analyzer.calls)) == (2, 0, 2)
    assert first.cost == pytest.approx(2 * (1000 * 1.0 + 200 * 5.0) / 1e6)
    second = run_analysis(jobs, analyzer, HAIKU, cache, max_cost=10)
    assert (second.analyzed, second.from_cache, second.cost) == (0, 2, 0.0)
    assert len(analyzer.calls) == 2                      # no new call to the model

    results = load_results(jobs, cache)
    assert sorted(results["ticker"]) == ["AIR.PA", "KER.PA", "MC.PA"]   # labels mapped back,
    assert set(results["prompt_version"]) == {"v1"}                     # out-of-list entry dropped
    assert results["anonymized"].all()


def test_cost_estimate_and_spending_cap(tmp_path):
    cache = AnalysisCache(tmp_path / "ai.sqlite")
    jobs = build_jobs(articles(), "claude-haiku-5-5")
    est = estimate_cost(jobs, cache, HAIKU)
    assert (est.n_articles, est.n_cached, est.n_to_analyze) == (2, 0, 2)
    assert est.cost == pytest.approx((est.input_tokens * 1.0 + est.output_tokens * 5.0) / 1e6)
    assert estimate_cost(jobs, cache, HAIKU, batch=True).cost == pytest.approx(est.cost / 2)

    analyzer = FakeAnalyzer()
    capped = run_analysis(jobs, analyzer, HAIKU, cache, max_cost=est.cost * 0.6)
    assert capped.stopped_by_cap and capped.analyzed == 1 and len(analyzer.calls) == 1
    assert estimate_cost(jobs, cache, HAIKU).n_cached == 1


def test_failed_article_does_not_stop_the_run(tmp_path):
    cache = AnalysisCache(tmp_path / "ai.sqlite")
    jobs = build_jobs(articles(), "claude-haiku-5-5", anonymized=False)
    report = run_analysis(jobs, FakeAnalyzer(fail_on="Airbus"), HAIKU, cache, max_cost=10)
    assert report.analyzed == 1 and list(report.failed.values()) == ["refus"]


def test_batch_mode(tmp_path):
    cache = AnalysisCache(tmp_path / "ai.sqlite")
    jobs = build_jobs(articles(), "claude-haiku-5-5")
    analyzer = FakeAnalyzer()
    report = run_analysis(jobs, analyzer, HAIKU, cache, max_cost=10, batch=True)
    assert report.analyzed == 2 and analyzer.calls == [] and len(analyzer.batch_calls[0]) == 2
    assert report.cost == pytest.approx(0.5 * 2 * (1000 * 1.0 + 200 * 5.0) / 1e6)


# -- Claude client (fake SDK client) -------------------------------------------------

def _message(payload, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=json.dumps(payload))],
                           usage=SimpleNamespace(input_tokens=321, output_tokens=45))


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.requests, self.stop_reason = [], stop_reason
        self.batches = FakeBatches()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _message({"analyses": [entry()]}, self.stop_reason)


class FakeBatches:
    def __init__(self):
        self.created, self.polls = [], 0

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="batch_1", processing_status="in_progress",
                               request_counts=SimpleNamespace(succeeded=0, errored=0, processing=2))

    def retrieve(self, batch_id):
        self.polls += 1
        return SimpleNamespace(id=batch_id, processing_status="ended", request_counts=None)

    def results(self, batch_id):
        yield SimpleNamespace(custom_id="k2", result=SimpleNamespace(type="errored"))
        yield SimpleNamespace(custom_id="k1", result=SimpleNamespace(
            type="succeeded", message=_message({"analyses": [entry()]})))


def test_claude_analyzer_request_and_parsing():
    messages = FakeMessages()
    analyzer = ClaudeAnalyzer("claude-haiku-5-5", client=SimpleNamespace(messages=messages))
    result = analyzer.analyze(SYSTEM, "Sociétés…")
    request = messages.requests[0]
    assert request["model"] == "claude-haiku-5-5" and request["system"] == SYSTEM
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert request["messages"] == [{"role": "user", "content": "Sociétés…"}]
    assert (result.input_tokens, result.output_tokens) == (321, 45)
    assert result.analyses[0].sentiment == 0.6


def test_claude_analyzer_refusal_is_an_article_error():
    client = SimpleNamespace(messages=FakeMessages(stop_reason="refusal"))
    with pytest.raises(AnalysisError, match="refusé"):
        ClaudeAnalyzer("claude-haiku-5-5", client=client).analyze(SYSTEM, "x")


def test_claude_batch_results_are_keyed_by_custom_id():
    messages = FakeMessages()
    analyzer = ClaudeAnalyzer("claude-haiku-5-5", client=SimpleNamespace(messages=messages),
                              sleep=lambda s: None)
    waits = []
    out = analyzer.analyze_batch(SYSTEM, {"k1": "u1", "k2": "u2"}, on_wait=waits.append)
    assert [r["custom_id"] for r in messages.batches.created] == ["k1", "k2"]
    assert messages.batches.created[0]["params"]["messages"][0]["content"] == "u1"
    assert isinstance(out["k1"], AnalyzerResult) and isinstance(out["k2"], AnalysisError)
    assert messages.batches.polls == 1 and len(waits) == 1


def test_missing_api_key_is_fatal(monkeypatch):
    from mugetrading.ai import claude_analyzer
    monkeypatch.setattr(claude_analyzer, "get_api_key", lambda name: "")
    with pytest.raises(FatalAnalysisError, match="Clé API"):
        ClaudeAnalyzer("claude-haiku-5-5").analyze(SYSTEM, "x")


def test_fatal_error_stops_the_run(tmp_path):
    class Broken(FakeAnalyzer):
        def analyze(self, system, user):
            raise FatalAnalysisError("Clé API Anthropic refusée.")
    report = run_analysis(build_jobs(articles(), "m"), Broken(), HAIKU,
                          AnalysisCache(tmp_path / "ai.sqlite"), max_cost=10)
    assert report.fatal_error and report.analyzed == 0


# -- local model ---------------------------------------------------------------------

def test_ollama_analyzer():
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        import json as j
        return SimpleNamespace(status_code=200, json=lambda: {
            "message": {"content": j.dumps({"analyses": [entry()]})},
            "prompt_eval_count": 50, "eval_count": 20})

    result = OllamaAnalyzer("llama3.1", http_post=fake_post).analyze(SYSTEM, "u")
    assert sent["url"] == "http://localhost:11434/api/chat" and sent["body"]["format"] == OUTPUT_SCHEMA
    assert (result.input_tokens, result.output_tokens) == (50, 20)

    def down(*a, **k):
        raise ConnectionError()
    with pytest.raises(FatalAnalysisError, match="Ollama injoignable"):
        OllamaAnalyzer("llama3.1", http_post=down).analyze(SYSTEM, "u")


def test_models_config():
    models, default = load_models()
    by_id = {m.id: m for m in models}
    assert default == "claude-haiku-5-5"
    assert {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-fable-5-2"} <= set(by_id)
    assert by_id["claude-haiku-5-5"].cost(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert any(m.provider == "ollama" and not m.supports_batch for m in models)
