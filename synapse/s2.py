"""Semantic Scholar Graph API client.

Cache-first by construction: `get()` consults sqlite before the network and
writes every successful response back. A cached paper is never re-fetched, which
is what lets a corpus build be interrupted, resumed, and replayed offline months
later and still produce the same graph.

Without an S2_API_KEY the shared pool 429s aggressively — that is the expected
condition here, not an error, so backoff is patient rather than fast-failing.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlencode

import requests

from .db import Database

log = logging.getLogger(__name__)

PAPER_FIELDS = (
    "paperId,corpusId,title,abstract,year,venue,publicationVenue,"
    "citationCount,referenceCount,s2FieldsOfStudy,externalIds,"
    "authors.authorId,authors.name"
)
# NOTE: the /references and /citations endpoints reject dotted author subfields
# ("Unrecognized or unsupported fields: [authors.name, authors.authorId]") even
# though /paper accepts them. Bare `authors` returns the same objects. Verified
# against the live API 2026-08-10 -- do not "tidy" this to match PAPER_FIELDS.
REFERENCE_FIELDS = (
    "paperId,corpusId,title,abstract,year,venue,citationCount,referenceCount,"
    "s2FieldsOfStudy,authors"
)


class RateLimited(Exception):
    """Raised when retries are exhausted against a 429/5xx."""


class FetchFailed(Exception):
    """The server did not answer. NOT the same as "the server said nothing".

    This distinction is the whole point of the class. Returning `[]` for both
    "this paper genuinely has no references" and "we gave up after ten 429s"
    is what let a corpus build once report success with zero citation edges.
    The existing `expansion_failed` guard only catches the total-zero case; a
    build where a third of the expansion calls quietly failed still looked
    healthy, and every downstream number would have been computed over a
    silently thinned graph.
    """


@dataclass
class FetchStats:
    """Counters surfaced in run artifacts so cache behaviour is auditable."""

    cache_hits: int = 0
    network_calls: int = 0
    rate_limit_waits: int = 0
    failures: int = 0
    total_wait_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "network_calls": self.network_calls,
            "rate_limit_waits": self.rate_limit_waits,
            "failures": self.failures,
            "total_wait_seconds": round(self.total_wait_seconds, 2),
        }


class SemanticScholarClient:
    def __init__(
        self,
        db: Database,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: str | None = None,
        requests_per_second: float = 0.7,
        max_retries: int = 6,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 90.0,
        timeout_seconds: int = 30,
        offline: bool = False,
        seed: int = 0,
    ):
        self.db = db
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.backoff_max = backoff_max_seconds
        self.timeout = timeout_seconds
        self.offline = offline
        self.stats = FetchStats()
        self._last_request_at = 0.0
        self._session = requests.Session()
        # Jitter is seeded so even the backoff schedule is reproducible.
        self._random = random.Random(seed or 0)

    # ------------------------------------------------------------------ core

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "synapse-ir/0.1 (research)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _cache_key(self, path: str, params: Mapping[str, Any]) -> str:
        canonical = f"{path}?{urlencode(sorted(params.items()))}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any | None:
        """GET with cache-first semantics.

        Returns the payload on 200, and None ONLY for a confirmed 404. Every
        other failure raises `FetchFailed`, so a caller can never mistake "the
        server refused" for "the resource is empty". See `FetchFailed`.
        """
        params = dict(params or {})
        cache_key = self._cache_key(path, params)

        cached = self.db.get_api_cache(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached["payload"] if cached["status"] == 200 else None

        if self.offline:
            raise FetchFailed(f"offline and {path} is not cached")

        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                self.stats.network_calls += 1
                response = self._session.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
            except requests.RequestException as exc:
                log.warning("network error on %s (attempt %d): %s", path, attempt + 1, exc)
                if attempt >= self.max_retries:
                    self.stats.failures += 1
                    raise FetchFailed(f"network error on {path}: {exc}") from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    self.stats.failures += 1
                    raise FetchFailed(f"malformed JSON from {path}") from exc
                self.db.put_api_cache(cache_key, path, payload, 200)
                return payload

            if response.status_code == 404:
                # Negative results are cached too: a paper that does not exist
                # should not be re-requested on every rebuild.
                self.db.put_api_cache(cache_key, path, None, 404)
                return None

            if response.status_code in (429, 500, 502, 503, 504):
                self.stats.rate_limit_waits += 1
                if attempt >= self.max_retries:
                    self.stats.failures += 1
                    log.warning("giving up on %s after %d retries (%d)",
                                path, self.max_retries, response.status_code)
                    raise FetchFailed(
                        f"{path} still returning {response.status_code} after "
                        f"{self.max_retries} retries"
                    )
                self._sleep_backoff(attempt, response)
                continue

            log.warning("unexpected status %d for %s", response.status_code, path)
            self.stats.failures += 1
            raise FetchFailed(f"unexpected status {response.status_code} for {path}")

        raise FetchFailed(f"exhausted retries for {path}")

    def _sleep_backoff(self, attempt: int, response: requests.Response | None = None) -> None:
        retry_after = None
        if response is not None:
            raw = response.headers.get("Retry-After")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        if retry_after is None:
            retry_after = self.backoff_base * (2 ** attempt)
        delay = min(retry_after, self.backoff_max)
        delay += self._random.uniform(0, min(1.0, delay * 0.1))
        self.stats.total_wait_seconds += delay
        log.info("backing off %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    # ----------------------------------------------------------- entity calls

    # These raise FetchFailed rather than returning []. An empty list from any of
    # them now means the API confirmed there is nothing, which is what makes the
    # per-paper failure accounting in `ingest.build_corpus` trustworthy.

    def paper(self, paper_id: str, fields: str = PAPER_FIELDS) -> dict[str, Any] | None:
        return self.get(f"paper/{paper_id}", {"fields": fields})

    def references(self, paper_id: str, limit: int = 60) -> list[dict[str, Any]]:
        payload = self.get(
            f"paper/{paper_id}/references",
            {"fields": REFERENCE_FIELDS, "limit": min(limit, 1000)},
        )
        if not payload:
            return []
        return [item["citedPaper"] for item in payload.get("data", []) if item.get("citedPaper")]

    def citations(self, paper_id: str, limit: int = 60) -> list[dict[str, Any]]:
        payload = self.get(
            f"paper/{paper_id}/citations",
            {"fields": REFERENCE_FIELDS, "limit": min(limit, 1000)},
        )
        if not payload:
            return []
        return [item["citingPaper"] for item in payload.get("data", []) if item.get("citingPaper")]

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        payload = self.get(
            "paper/search", {"query": query, "limit": limit, "fields": PAPER_FIELDS}
        )
        if not payload:
            return []
        return payload.get("data", [])


def normalise_paper(raw: Mapping[str, Any], hop: int | None = None) -> dict[str, Any] | None:
    """Flatten an S2 record into our storage shape.

    Returns None for records with no usable identifier — the corpus is keyed on
    paperId and a record without one cannot be cited, ablated, or explained.
    """
    paper_id = raw.get("paperId")
    if not paper_id:
        return None

    venue = raw.get("venue") or ""
    publication_venue = raw.get("publicationVenue")
    if not venue and isinstance(publication_venue, Mapping):
        venue = publication_venue.get("name") or ""

    fields: list[str] = []
    for entry in raw.get("s2FieldsOfStudy") or []:
        if isinstance(entry, Mapping):
            category = entry.get("category")
            if category and category not in fields:
                fields.append(category)

    authors = []
    for author in raw.get("authors") or []:
        if isinstance(author, Mapping) and author.get("authorId"):
            authors.append(
                {"author_id": str(author["authorId"]), "name": author.get("name") or ""}
            )

    return {
        "paper_id": str(paper_id),
        "corpus_id": str(raw["corpusId"]) if raw.get("corpusId") else None,
        "title": (raw.get("title") or "").strip(),
        "abstract": (raw.get("abstract") or "").strip(),
        "year": raw.get("year"),
        "venue": venue.strip(),
        "citation_count": raw.get("citationCount") or 0,
        "reference_count": raw.get("referenceCount") or 0,
        "fields": fields,
        "authors": authors,
        "source": "s2",
        "hop": hop,
    }
