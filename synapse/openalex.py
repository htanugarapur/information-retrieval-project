"""OpenAlex ingest source.

WHY THIS EXISTS (this belongs in the paper's reproducibility section)
---------------------------------------------------------------------
Phase 1 specifies Semantic Scholar. On 2026-08-10 the unauthenticated S2 Graph
API returned 429 on every request from this host after an initial burst, making
a 500-paper neighbourhood expansion impossible without an API key. OpenAlex
serves the same graph structure -- citations, authorships, venues, topics --
under a generous open rate limit and no key.

Both sources normalise into the IDENTICAL schema and produce the identical typed
edges, so nothing downstream of `synapse.db` can tell them apart. Papers carry
`source` ('s2' or 'openalex'), and every run artifact records which source built
the corpus. The choice is a data-availability fact, not a methodological one, but
it is recorded rather than hidden because corpus provenance changes what a
result generalises to.

API notes verified live 2026-08-10:
  * `title` is often null; `display_name` carries the title.
  * abstracts ship as `abstract_inverted_index` and must be reconstructed.
  * incoming citations come from `/works?filter=cites:<id>`, not a field.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import requests

from .db import Database
from .s2 import FetchFailed, FetchStats

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"

# Requested explicitly so the payload stays small and the cache stays comparable
# across runs; OpenAlex returns a very large default record otherwise.
WORK_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "abstract_inverted_index",
        "cited_by_count",
        "referenced_works",
        "authorships",
        "primary_location",
        "topics",
        "type",
    ]
)


def short_id(openalex_id: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'. Ids are stored short and stable."""
    if not openalex_id:
        return None
    return str(openalex_id).rsplit("/", 1)[-1]


def _normalise_title(title: str) -> str:
    """Casefold and strip punctuation/whitespace for exact-title comparison."""
    return "".join(ch for ch in (title or "").lower() if ch.isalnum())


def reconstruct_abstract(inverted_index: Mapping[str, Sequence[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's inverted index.

    The index maps token -> list of positions. Rebuilding is exact apart from
    whitespace normalisation, which is all BM25 and the dense encoder need.
    """
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, indices in inverted_index.items():
        for position in indices or []:
            try:
                positions.append((int(position), token))
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    positions.sort()
    return " ".join(token for _, token in positions)


class OpenAlexClient:
    """Cache-first OpenAlex client, mirroring SemanticScholarClient's contract."""

    def __init__(
        self,
        db: Database,
        base_url: str = BASE_URL,
        mailto: str | None = None,
        requests_per_second: float = 5.0,
        max_retries: int = 5,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 60.0,
        timeout_seconds: int = 30,
        offline: bool = False,
        seed: int = 0,
    ):
        self.db = db
        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.backoff_max = backoff_max_seconds
        self.timeout = timeout_seconds
        self.offline = offline
        self.stats = FetchStats()
        self._last_request_at = 0.0
        self._session = requests.Session()
        self._random = random.Random(seed)

    def _cache_key(self, path: str, params: Mapping[str, Any]) -> str:
        import hashlib

        canonical = f"openalex:{path}?{urlencode(sorted(params.items()))}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any | None:
        """Payload on 200, None ONLY on a confirmed 404, `FetchFailed` otherwise.

        Mirrors `SemanticScholarClient.get` deliberately: a caller must never be
        able to mistake "the server refused" for "there is nothing here".
        """
        params = dict(params or {})
        if self.mailto:
            # The "polite pool" is faster and more reliable, and identifying the
            # caller is the courtesy OpenAlex asks for in exchange for open access.
            params.setdefault("mailto", self.mailto)

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
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("network error on %s: %s", path, exc)
                if attempt >= self.max_retries:
                    self.stats.failures += 1
                    raise FetchFailed(f"network error on {path}: {exc}") from exc
                self._sleep(attempt)
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
                self.db.put_api_cache(cache_key, path, None, 404)
                return None

            if response.status_code in (429, 500, 502, 503, 504):
                self.stats.rate_limit_waits += 1
                if attempt >= self.max_retries:
                    self.stats.failures += 1
                    raise FetchFailed(
                        f"{path} still returning {response.status_code} after "
                        f"{self.max_retries} retries"
                    )
                self._sleep(attempt)
                continue

            log.warning("unexpected status %d for %s: %s",
                        response.status_code, path, response.text[:200])
            self.stats.failures += 1
            raise FetchFailed(f"unexpected status {response.status_code} for {path}")

        raise FetchFailed(f"exhausted retries for {path}")

    def _sleep(self, attempt: int) -> None:
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        delay += self._random.uniform(0, 1.0)
        self.stats.total_wait_seconds += delay
        time.sleep(delay)

    # ------------------------------------------------------------- entities

    def work(self, work_id: str) -> dict[str, Any] | None:
        """Fetch one work. Accepts W-ids, DOIs, 'arxiv:ID', and 'title:Exact Title'.

        The arXiv-DOI route resolves only preprints that OpenAlex indexed under
        the 10.48550 prefix. Papers that were later published are indexed under
        the publisher DOI instead and 404 on the arXiv form -- DPR and SPECTER
        both do. Hence the explicit `title:` form, which requires an EXACT
        case-insensitive title match: a fuzzy search would happily return
        RocketQA when asked for DPR, and silently seeding the wrong paper would
        corrupt every downstream result.
        """
        identifier = work_id.strip()

        if identifier.lower().startswith("title:"):
            return self._work_by_exact_title(identifier.split(":", 1)[1].strip())

        if identifier.lower().startswith("arxiv:"):
            arxiv_id = identifier.split(":", 1)[1]
            found = self.get(
                f"works/https://doi.org/10.48550/arXiv.{arxiv_id}", {"select": WORK_FIELDS}
            )
            if found:
                return found
            log.info("arXiv DOI lookup failed for %s; trying arxiv id search", arxiv_id)
            return self._work_by_arxiv_search(arxiv_id)

        if identifier.startswith("10."):
            identifier = f"https://doi.org/{identifier}"
        return self.get(f"works/{identifier}", {"select": WORK_FIELDS})

    def _work_by_exact_title(self, title: str) -> dict[str, Any] | None:
        payload = self.get(
            "works",
            {"filter": f"title.search:{title}", "per-page": 25, "select": WORK_FIELDS},
        )
        target = _normalise_title(title)
        for candidate in (payload or {}).get("results") or []:
            name = candidate.get("title") or candidate.get("display_name") or ""
            if _normalise_title(name) == target:
                return candidate
        log.warning("no exact title match for %r", title)
        return None

    def _work_by_arxiv_search(self, arxiv_id: str) -> dict[str, Any] | None:
        payload = self.get(
            "works",
            {"filter": f"locations.landing_page_url.search:{arxiv_id}",
             "per-page": 5, "select": WORK_FIELDS},
        )
        results = (payload or {}).get("results") or []
        return results[0] if results else None

    def works_batch(
        self, work_ids: Sequence[str], per_page: int = 50
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch many works via an OR filter, 50 ids per call.

        Returns (works, failed_id_count). A failed chunk previously vanished
        silently, taking up to 50 papers' worth of citation edges with it and
        leaving no trace anywhere in the artifact.
        """
        results: list[dict[str, Any]] = []
        failed = 0
        ids = [short_id(i) for i in work_ids if i]
        for start in range(0, len(ids), per_page):
            chunk = [i for i in ids[start : start + per_page] if i]
            if not chunk:
                continue
            try:
                payload = self.get(
                    "works",
                    {
                        "filter": f"openalex_id:{'|'.join(chunk)}",
                        "per-page": len(chunk),
                        "select": WORK_FIELDS,
                    },
                )
            except FetchFailed as exc:
                log.warning("batch of %d works failed: %s", len(chunk), exc)
                failed += len(chunk)
                continue
            if payload:
                results.extend(payload.get("results") or [])
        return results, failed

    def citing_works(self, work_id: str, limit: int = 25) -> list[dict[str, Any]]:
        """Works that cite this one."""
        payload = self.get(
            "works",
            {
                "filter": f"cites:{short_id(work_id)}",
                "per-page": min(limit, 200),
                "sort": "cited_by_count:desc",
                "select": WORK_FIELDS,
            },
        )
        return (payload or {}).get("results") or []

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        payload = self.get(
            "works",
            {"search": query, "per-page": min(limit, 200), "select": WORK_FIELDS},
        )
        return (payload or {}).get("results") or []


def normalise_work(raw: Mapping[str, Any], hop: int | None = None) -> dict[str, Any] | None:
    """Flatten an OpenAlex work into the shared storage schema."""
    work_id = short_id(raw.get("id"))
    if not work_id:
        return None

    venue = ""
    primary_location = raw.get("primary_location") or {}
    if isinstance(primary_location, Mapping):
        source = primary_location.get("source") or {}
        if isinstance(source, Mapping):
            venue = source.get("display_name") or ""

    fields: list[str] = []
    for topic in raw.get("topics") or []:
        if isinstance(topic, Mapping):
            name = topic.get("display_name")
            if name and name not in fields:
                fields.append(name)

    authors = []
    for authorship in raw.get("authorships") or []:
        if not isinstance(authorship, Mapping):
            continue
        author = authorship.get("author") or {}
        author_id = short_id(author.get("id")) if isinstance(author, Mapping) else None
        if author_id:
            authors.append(
                {
                    "author_id": author_id,
                    "name": (author.get("display_name") if isinstance(author, Mapping) else "") or "",
                }
            )

    references = [short_id(r) for r in (raw.get("referenced_works") or [])]

    return {
        "paper_id": work_id,
        "corpus_id": work_id,
        # `title` is frequently null in OpenAlex; display_name is the reliable field.
        "title": (raw.get("title") or raw.get("display_name") or "").strip(),
        "abstract": reconstruct_abstract(raw.get("abstract_inverted_index")),
        "year": raw.get("publication_year"),
        "venue": venue.strip(),
        "citation_count": raw.get("cited_by_count") or 0,
        "reference_count": len(references),
        "fields": fields,
        "authors": authors,
        "references": [r for r in references if r],
        "source": "openalex",
        "hop": hop,
    }
