"""Tests for the HTTP clients: caching, backoff, and the LLM fallback chain.

These are not incidental plumbing. Cache-first behaviour is what makes a run
reproducible and offline-replayable, and the LLM model-fallback is what stops a
mixed-model result being reported as one number. Both are tested against fake
transports rather than the live APIs so the suite is deterministic and offline.
"""

from __future__ import annotations

import json

import pytest
import requests

from synapse.db import Database
from synapse.ingest import MAX_EXPANSION_FAILURE_RATE, CorpusStats, build_corpus
from synapse.llm import LLMUnavailable, OpenRouterClient, parse_json_object
from synapse.openalex import OpenAlexClient
from synapse.s2 import FetchFailed, SemanticScholarClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload if payload is not None else {})
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Replays a queued script of responses and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def _next(self):
        if not self.script:
            return FakeResponse(500)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return self._next()

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json or {}, "headers": headers or {}})
        return self._next()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "clients.sqlite")
    yield database
    database.close()


def s2_client(db, script, **kwargs):
    client = SemanticScholarClient(db, requests_per_second=0, **kwargs)
    client._session = FakeSession(script)
    return client


# ------------------------------------------------------------ S2: caching


def test_successful_response_is_cached_and_not_refetched(db):
    client = s2_client(db, [FakeResponse(200, {"paperId": "p1", "title": "T"})])

    first = client.paper("p1")
    second = client.paper("p1")

    assert first == second == {"paperId": "p1", "title": "T"}
    assert len(client._session.calls) == 1
    assert client.stats.cache_hits == 1


def test_a_404_is_cached_so_it_is_never_requested_twice(db):
    client = s2_client(db, [FakeResponse(404)])

    assert client.paper("missing") is None
    assert client.paper("missing") is None
    assert len(client._session.calls) == 1


def test_api_key_is_sent_as_a_header_when_present(db):
    client = s2_client(db, [FakeResponse(200, {"paperId": "p"})], api_key="secret")

    client.paper("p")

    assert client._session.calls[0]["headers"]["x-api-key"] == "secret"


def test_no_api_key_header_when_none_configured(db):
    client = s2_client(db, [FakeResponse(200, {"paperId": "p"})])

    client.paper("p")

    assert "x-api-key" not in client._session.calls[0]["headers"]


def test_offline_client_raises_rather_than_reporting_an_empty_result(db):
    """Offline-and-uncached is a failure, not a confirmed absence."""
    client = s2_client(db, [FakeResponse(200, {"paperId": "p"})])
    client.offline = True

    with pytest.raises(FetchFailed):
        client.paper("p")
    assert client._session.calls == []


def test_offline_client_still_serves_cached_records(db):
    client = s2_client(db, [FakeResponse(200, {"paperId": "p"})])
    client.paper("p")           # populate the cache
    client.offline = True

    assert client.paper("p") == {"paperId": "p"}


# ------------------------------------------------------------ S2: retries


def test_429_is_retried_then_succeeds(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = s2_client(
        db, [FakeResponse(429), FakeResponse(429), FakeResponse(200, {"paperId": "p"})],
        max_retries=5, backoff_base_seconds=0.01,
    )

    assert client.paper("p") == {"paperId": "p"}
    assert client.stats.rate_limit_waits == 2


def test_retries_are_bounded_then_raise(db, monkeypatch):
    """Exhausted retries must be distinguishable from "this paper has nothing"."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = s2_client(db, [FakeResponse(429)] * 10, max_retries=2, backoff_base_seconds=0.01)

    with pytest.raises(FetchFailed, match="429"):
        client.paper("p")
    assert client.stats.failures == 1


def test_a_400_is_not_retried(db):
    """A malformed request will never succeed; retrying it just wastes the pool."""
    client = s2_client(db, [FakeResponse(400, text="bad fields"), FakeResponse(200, {"x": 1})])

    with pytest.raises(FetchFailed, match="400"):
        client.paper("p")
    assert len(client._session.calls) == 1


def test_a_404_is_a_confirmed_absence_not_a_failure(db):
    """The one case that legitimately returns None -- the server answered."""
    client = s2_client(db, [FakeResponse(404)])

    assert client.paper("gone") is None


def test_references_returning_empty_means_the_api_confirmed_empty(db):
    """The distinction the per-paper failure accounting in ingest depends on."""
    client = s2_client(db, [FakeResponse(200, {"data": []})])

    assert client.references("p") == []


def test_network_errors_are_retried(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = s2_client(
        db, [requests.RequestException("boom"), FakeResponse(200, {"paperId": "p"})],
        max_retries=3, backoff_base_seconds=0.01,
    )

    assert client.paper("p") == {"paperId": "p"}


def test_retry_after_header_is_respected(db, monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    client = s2_client(
        db, [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"p": 1})],
        max_retries=3, backoff_max_seconds=60,
    )

    client.paper("p")

    assert slept and 7 <= slept[0] <= 8


def test_backoff_is_reproducible_for_a_fixed_seed(db, monkeypatch):
    """Jitter is seeded, so even the retry schedule replays identically."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    def schedule(seed):
        slept.clear()
        client = s2_client(db, [], max_retries=3, backoff_base_seconds=1.0, seed=seed)
        for attempt in range(3):
            client._sleep_backoff(attempt)
        return list(slept)

    assert schedule(42) == schedule(42)
    assert schedule(42) != schedule(7)


def test_references_are_unwrapped_from_the_cited_paper_envelope(db):
    payload = {"data": [{"citedPaper": {"paperId": "c1"}}, {"citedPaper": None}]}
    client = s2_client(db, [FakeResponse(200, payload)])

    assert client.references("p") == [{"paperId": "c1"}]


def test_citations_are_unwrapped_from_the_citing_paper_envelope(db):
    payload = {"data": [{"citingPaper": {"paperId": "x1"}}]}
    client = s2_client(db, [FakeResponse(200, payload)])

    assert client.citations("p") == [{"paperId": "x1"}]


def test_reference_fields_do_not_use_dotted_author_subfields(db):
    """The live API rejects authors.authorId here; regression guard."""
    client = s2_client(db, [FakeResponse(200, {"data": []})])

    client.references("p")

    assert "authors.authorId" not in client._session.calls[0]["params"]["fields"]


# ------------------------------------------------------------- OpenAlex


def openalex_client(db, script, **kwargs):
    client = OpenAlexClient(db, requests_per_second=0, **kwargs)
    client._session = FakeSession(script)
    return client


def test_openalex_adds_the_polite_pool_mailto(db):
    client = openalex_client(db, [FakeResponse(200, {"id": "W1"})], mailto="a@b.c")

    client.work("W1")

    assert client._session.calls[0]["params"]["mailto"] == "a@b.c"


def test_openalex_caches_responses(db):
    client = openalex_client(db, [FakeResponse(200, {"id": "W1"})])

    client.work("W1")
    client.work("W1")

    assert len(client._session.calls) == 1


def test_arxiv_id_falls_back_to_search_when_the_doi_404s(db):
    """DPR and SPECTER both 404 on their arXiv DOI; the fallback must catch that."""
    client = openalex_client(db, [
        FakeResponse(404),
        FakeResponse(200, {"results": [{"id": "https://openalex.org/W99"}]}),
    ])

    assert client.work("arXiv:2004.04906") == {"id": "https://openalex.org/W99"}


def test_title_lookup_requires_an_exact_match(db):
    """A fuzzy search returns RocketQA when asked for DPR; that must be rejected."""
    client = openalex_client(db, [FakeResponse(200, {"results": [
        {"id": "https://openalex.org/W1", "display_name": "RocketQA: An Optimized Approach"},
    ]})])

    assert client.work("title:Dense Passage Retrieval") is None


def test_title_lookup_accepts_a_match_differing_only_in_case_and_punctuation(db):
    client = openalex_client(db, [FakeResponse(200, {"results": [
        {"id": "https://openalex.org/W2", "display_name": "Dense Passage Retrieval!"},
    ]})])

    assert client.work("title:dense passage retrieval")["id"].endswith("W2")


def test_batch_fetch_chunks_ids(db):
    client = openalex_client(db, [FakeResponse(200, {"results": [{"id": "W1"}]}),
                                  FakeResponse(200, {"results": [{"id": "W2"}]})])

    results = client.works_batch([f"W{i}" for i in range(3)], per_page=2)

    assert len(client._session.calls) == 2
    assert len(results) == 2


def test_citing_works_filters_on_cites(db):
    client = openalex_client(db, [FakeResponse(200, {"results": []})])

    client.citing_works("https://openalex.org/W5")

    assert client._session.calls[0]["params"]["filter"] == "cites:W5"


# ---------------------------------------------------------------- OpenRouter


def llm_client(db, script, **kwargs):
    client = OpenRouterClient(db, api_key="k", **kwargs)
    client._session = FakeSession(script)
    return client


def completion(text, model="m1"):
    return FakeResponse(200, {"choices": [{"message": {"content": text}}],
                              "model": model, "usage": {"total_tokens": 10}})


def test_completion_is_cached_and_replayed_offline(db):
    client = llm_client(db, [completion("hello")])

    first = client.complete("prompt")
    second = client.complete("prompt")

    assert first.text == second.text == "hello"
    assert second.cached is True
    assert len(client._session.calls) == 1


def test_missing_api_key_raises_rather_than_degrading(db):
    client = OpenRouterClient(db, api_key=None)

    with pytest.raises(LLMUnavailable, match="OPENROUTER_API_KEY"):
        client.complete("prompt")


def test_client_reports_it_is_unavailable_without_a_key(db):
    assert OpenRouterClient(db, api_key=None).available is False


def test_fallback_chain_moves_to_the_next_model_on_429(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = llm_client(
        db, [FakeResponse(429), FakeResponse(429), completion("ok", model="m2")],
        model="m1", fallback_models=["m2"], max_retries=2,
    )

    response = client.complete("p")

    assert response.text == "ok"


def test_response_records_which_model_actually_answered(db):
    """A mixed-model result reported as one number would be a lie."""
    client = llm_client(db, [completion("ok", model="served-by-this")])

    assert client.complete("p").model == "served-by-this"


def test_a_400_skips_straight_to_the_next_model(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = llm_client(
        db, [FakeResponse(400, text="bad model"), completion("ok", model="m2")],
        model="m1", fallback_models=["m2"], max_retries=4,
    )

    assert client.complete("p").text == "ok"
    assert len(client._session.calls) == 2


def test_exhausting_every_model_raises(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = llm_client(db, [FakeResponse(429)] * 12, model="m1",
                        fallback_models=["m2"], max_retries=1)

    with pytest.raises(LLMUnavailable, match="all models failed"):
        client.complete("p")


def test_complete_json_returns_none_for_unparseable_output(db):
    client = llm_client(db, [completion("not json")])

    parsed, response = client.complete_json("p")

    assert parsed is None and response.text == "not json"


def test_complete_json_parses_a_fenced_block(db):
    client = llm_client(db, [completion('```json\n{"a": 1}\n```')])

    parsed, _ = client.complete_json("p")

    assert parsed == {"a": 1}


def test_free_model_listing_filters_on_zero_price(db):
    client = llm_client(db, [FakeResponse(200, {"data": [
        {"id": "free-1", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 100},
        {"id": "paid-1", "pricing": {"prompt": "0.5", "completion": "1"}, "context_length": 900},
    ]})])

    free = client.list_free_models()

    assert [m["id"] for m in free] == ["free-1"]


def test_model_listing_failure_returns_empty_rather_than_raising(db):
    client = llm_client(db, [requests.RequestException("down")])

    assert client.list_free_models() == []


def test_model_chain_puts_the_primary_first_without_duplicates(db):
    client = OpenRouterClient(db, api_key="k", model="a", fallback_models=["a", "b"])

    assert client.model_chain() == ["a", "b"]


# ------------------------------------------------- partial-degradation guard
#
# The build once reported SUCCESS having produced zero citation edges. The guard
# added then only caught the total-zero case; these cover the far likelier
# partial case, where enough calls succeed to look healthy while the graph every
# downstream number is computed over is materially thinner than requested.


def test_expansion_failure_rate_counts_failed_calls():
    stats = CorpusStats(papers_expanded=90, expansion_calls_failed=10)

    assert stats.expansion_failure_rate == pytest.approx(0.10)


def test_expansion_failure_rate_is_zero_when_nothing_was_attempted():
    assert CorpusStats().expansion_failure_rate == 0.0


def _seed_paper(pid="p1"):
    return FakeResponse(200, {"paperId": pid, "title": "T", "abstract": "a" * 100,
                              "year": 2020, "venue": "V", "authors": []})


def test_a_partially_failed_build_is_flagged_as_degraded(db, monkeypatch):
    """Some edges land, so the old all-zero check passes — this must still fail."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    script = [
        _seed_paper("p1"),
        # references succeed, citations exhaust retries
        FakeResponse(200, {"data": [{"citedPaper": {"paperId": "p2", "title": "B",
                                                    "abstract": "b" * 100, "year": 2018,
                                                    "authors": []}}]}),
        FakeResponse(429), FakeResponse(429), FakeResponse(429),
    ]
    client = s2_client(db, script, max_retries=2, backoff_base_seconds=0.01)

    stats = build_corpus(db, client, ["p1"], hops=1, max_papers=10, progress=False)

    assert stats.edges_by_relation.get("cites", 0) > 0   # looks healthy
    assert stats.expansion_calls_failed == 1
    assert stats.expansion_degraded is True              # but is not
    assert not stats.expansion_failed                    # the old guard misses it


def test_a_clean_build_is_not_flagged_as_degraded(db):
    script = [
        _seed_paper("p1"),
        FakeResponse(200, {"data": [{"citedPaper": {"paperId": "p2", "title": "B",
                                                    "abstract": "b" * 100, "year": 2018,
                                                    "authors": []}}]}),
        FakeResponse(200, {"data": []}),
    ]
    client = s2_client(db, script)

    stats = build_corpus(db, client, ["p1"], hops=1, max_papers=10, progress=False)

    assert stats.expansion_calls_failed == 0
    assert stats.expansion_degraded is False
    assert stats.papers_expanded == 1


def test_a_seed_that_could_not_be_fetched_is_not_recorded_as_missing_only(db, monkeypatch):
    """A fetch failure must not be laundered into "this paper does not exist"."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = s2_client(db, [FakeResponse(429)] * 6, max_retries=2, backoff_base_seconds=0.01)

    stats = build_corpus(db, client, ["p1"], hops=1, max_papers=10, progress=False)

    assert stats.seeds_missing == ["p1"]
    assert stats.expansion_calls_failed == 1


def test_degradation_threshold_is_the_documented_constant():
    assert 0 < MAX_EXPANSION_FAILURE_RATE < 1
