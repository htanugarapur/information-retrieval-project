"""Phase 1 — corpus construction and the typed graph.

`build_corpus` expands a citation neighbourhood breadth-first from seed papers
and persists nodes plus typed edges. Concept nodes come from two sources that
are deliberately kept distinguishable in the edge attrs (`origin`): S2's own
`s2FieldsOfStudy`, and YAKE keyphrases over title+abstract.

Keeping the origin on the edge matters for the ablation: if a result turns out to
be carried entirely by coarse S2 field edges rather than by specific keyphrase
edges, that is a finding about the instrument, and it is only visible if the two
were never merged.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from .db import Database
from .s2 import FetchFailed, SemanticScholarClient, normalise_paper

if TYPE_CHECKING:  # import cycle at runtime: openalex imports from s2, s2 from db
    from .openalex import OpenAlexClient

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9 \-]")

# Above this share of failed neighbourhood expansions the corpus is reported as
# degraded. Not zero: on a shared API pool a few failures are normal and the
# sqlite cache makes them recoverable by re-running. But a build that loses more
# than a twentieth of its expansions has a materially different graph from the
# one its parameters describe, and that must not pass silently.
MAX_EXPANSION_FAILURE_RATE = 0.05


# ------------------------------------------------------------------ identity


def slugify(text: str) -> str:
    """Stable id fragment. Deterministic across machines and Python versions."""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = _NON_WORD.sub(" ", ascii_only)
    return _WHITESPACE.sub("-", cleaned.strip())[:80]


def author_node_id(author_id: str) -> str:
    return f"author:{author_id}"


def venue_node_id(venue_name: str) -> str:
    return f"venue:{slugify(venue_name)}"


def concept_node_id(concept: str) -> str:
    return f"concept:{slugify(concept)}"


# ------------------------------------------------------------------- summary


@dataclass
class CorpusStats:
    papers: int = 0
    authors: int = 0
    venues: int = 0
    concepts: int = 0
    edges_by_relation: dict[str, int] = field(default_factory=dict)
    seeds_resolved: list[str] = field(default_factory=list)
    seeds_missing: list[str] = field(default_factory=list)
    fetch: dict[str, Any] = field(default_factory=dict)
    frontier_truncated: bool = False
    expansion_failed: bool = False

    # Per-paper expansion accounting. `expansion_failed` alone only catches a
    # corpus with ZERO citation edges; these catch the far more likely case of a
    # build where some fraction of expansion calls quietly gave up, producing a
    # thinner graph that still looks healthy. Every downstream number is computed
    # over this graph, so the degradation rate belongs in the artifact.
    papers_expanded: int = 0
    expansion_calls_failed: int = 0
    expansion_degraded: bool = False

    @property
    def expansion_failure_rate(self) -> float:
        attempted = self.papers_expanded + self.expansion_calls_failed
        return self.expansion_calls_failed / attempted if attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "authors": self.authors,
            "venues": self.venues,
            "concepts": self.concepts,
            "edges_by_relation": self.edges_by_relation,
            "seeds_resolved": self.seeds_resolved,
            "seeds_missing": self.seeds_missing,
            "fetch": self.fetch,
            "frontier_truncated": self.frontier_truncated,
            "expansion_failed": self.expansion_failed,
            "papers_expanded": self.papers_expanded,
            "expansion_calls_failed": self.expansion_calls_failed,
            "expansion_failure_rate": round(self.expansion_failure_rate, 4),
            "expansion_degraded": self.expansion_degraded,
        }


# -------------------------------------------------------------- construction


def build_corpus(
    db: Database,
    client: SemanticScholarClient,
    seed_ids: Sequence[str],
    hops: int = 2,
    max_papers: int = 5000,
    max_refs_per_paper: int = 60,
    max_cites_per_paper: int = 60,
    progress: bool = True,
) -> CorpusStats:
    """Breadth-first expansion of a citation neighbourhood.

    Papers are persisted as they arrive rather than at the end, so an
    interrupted build (very likely on the unauthenticated S2 pool) resumes from
    the sqlite cache instead of starting over.
    """
    stats = CorpusStats()
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for seed in seed_ids:
        try:
            raw = client.paper(seed)
        except FetchFailed as exc:
            # A seed that could not be FETCHED is not the same as a seed that
            # does not exist, and conflating them would silently shrink the
            # experiment's scope.
            log.error("seed fetch failed for %s: %s", seed, exc)
            stats.seeds_missing.append(seed)
            stats.expansion_calls_failed += 1
            continue
        if not raw:
            stats.seeds_missing.append(seed)
            log.warning("seed not resolved: %s", seed)
            continue
        paper = normalise_paper(raw, hop=0)
        if paper is None:
            stats.seeds_missing.append(seed)
            continue
        _persist_paper(db, paper)
        seen.add(paper["paper_id"])
        stats.seeds_resolved.append(paper["paper_id"])
        queue.append((paper["paper_id"], 0))

    if not stats.seeds_resolved:
        log.error("no seeds resolved; corpus not built")
        stats.fetch = client.stats.to_dict()
        return stats

    while queue:
        if len(seen) >= max_papers:
            stats.frontier_truncated = True
            log.info("hit max_papers=%d, stopping expansion", max_papers)
            break

        paper_id, hop = queue.popleft()
        if hop >= hops:
            continue

        # A failed expansion is COUNTED, not swallowed. Previously both endpoints
        # returned [] on failure, indistinguishable from a paper with no
        # references, so a partially-failed build looked identical to a complete
        # one.
        neighbours: list[tuple[dict[str, Any], str]] = []
        expansion_ok = True
        try:
            for raw in client.references(paper_id, limit=max_refs_per_paper):
                neighbours.append((raw, "outgoing"))
        except FetchFailed as exc:
            log.warning("references failed for %s: %s", paper_id, exc)
            expansion_ok = False
        try:
            for raw in client.citations(paper_id, limit=max_cites_per_paper):
                neighbours.append((raw, "incoming"))
        except FetchFailed as exc:
            log.warning("citations failed for %s: %s", paper_id, exc)
            expansion_ok = False

        if expansion_ok:
            stats.papers_expanded += 1
        else:
            stats.expansion_calls_failed += 1

        for raw, direction in neighbours:
            neighbour = normalise_paper(raw, hop=hop + 1)
            if neighbour is None:
                continue
            neighbour_id = neighbour["paper_id"]

            if neighbour_id not in seen:
                if len(seen) >= max_papers:
                    stats.frontier_truncated = True
                    continue
                _persist_paper(db, neighbour)
                seen.add(neighbour_id)
                if hop + 1 < hops:
                    queue.append((neighbour_id, hop + 1))

            # Citation direction is preserved: `cites` always points from the
            # citing paper to the cited one, regardless of which endpoint we
            # happened to be expanding from.
            if direction == "outgoing":
                db.add_edge(paper_id, "cites", neighbour_id)
            else:
                db.add_edge(neighbour_id, "cites", paper_id)

        if progress and len(seen) % 50 == 0:
            log.info("corpus: %d papers", len(seen))

    stats.fetch = client.stats.to_dict()
    return _finalise(db, stats)


def _persist_paper(db: Database, paper: Mapping[str, Any]) -> None:
    """Write a paper plus its author and venue nodes and edges."""
    db.upsert_paper(dict(paper))
    paper_id = paper["paper_id"]

    for author in paper.get("authors") or []:
        node_id = author_node_id(author["author_id"])
        db.upsert_node(node_id, "author", author.get("name") or author["author_id"])
        db.add_edge(paper_id, "authored_by", node_id)

    venue = (paper.get("venue") or "").strip()
    if venue:
        node_id = venue_node_id(venue)
        db.upsert_node(node_id, "venue", venue)
        db.add_edge(paper_id, "published_in", node_id)


def _finalise(db: Database, stats: CorpusStats) -> CorpusStats:
    stats.papers = db.count_nodes("paper")
    stats.authors = db.count_nodes("author")
    stats.venues = db.count_nodes("venue")
    stats.concepts = db.count_nodes("concept")
    stats.edges_by_relation = db.edge_counts_by_relation()

    # A corpus with no citation edges is not a corpus -- PPR would have nothing
    # to propagate along and every downstream number would be vacuous. This exact
    # failure happened once (the S2 API rejected a `fields` parameter, every
    # expansion call 400'd, and the build still reported "success"), so it is now
    # a hard, loud condition rather than a quiet zero.
    if stats.seeds_resolved and stats.edges_by_relation.get("cites", 0) == 0:
        stats.expansion_failed = True
        log.error(
            "NO CITATION EDGES were created from %d resolved seeds. Neighbourhood "
            "expansion failed -- fetch failures: %s. Check the log for HTTP errors.",
            len(stats.seeds_resolved),
            stats.fetch.get("failures"),
        )

    # The partial case, which the all-zero check above cannot see: enough calls
    # succeeded to produce a plausible-looking graph, but enough failed that the
    # graph is materially thinner than the parameters claim.
    if stats.expansion_failure_rate > MAX_EXPANSION_FAILURE_RATE:
        stats.expansion_degraded = True
        log.error(
            "DEGRADED CORPUS: %.1f%% of neighbourhood expansions failed (%d of %d). "
            "The citation graph is thinner than requested and every downstream "
            "number would be computed over it. Re-run to use the cache and fill "
            "the gaps before trusting results.",
            stats.expansion_failure_rate * 100,
            stats.expansion_calls_failed,
            stats.papers_expanded + stats.expansion_calls_failed,
        )
    return stats


def build_corpus_openalex(
    db: Database,
    client: "OpenAlexClient",
    seed_ids: Sequence[str],
    hops: int = 2,
    max_papers: int = 5000,
    max_refs_per_paper: int = 25,
    max_cites_per_paper: int = 25,
    progress: bool = True,
) -> CorpusStats:
    """Breadth-first expansion over OpenAlex.

    Structurally the same as `build_corpus`, but references arrive inline on the
    record (`referenced_works`) rather than from a separate endpoint, so they are
    batch-fetched 50 at a time. That makes a 500-paper build minutes rather than
    hours -- the reason this path exists at all (see synapse/openalex.py).
    """
    from .openalex import normalise_work

    stats = CorpusStats()
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    pending_citations: list[tuple[str, str]] = []

    for seed in seed_ids:
        try:
            raw = client.work(seed)
        except FetchFailed as exc:
            log.error("seed fetch failed for %s: %s", seed, exc)
            stats.seeds_missing.append(seed)
            stats.expansion_calls_failed += 1
            continue
        if not raw:
            stats.seeds_missing.append(seed)
            log.warning("seed not resolved: %s", seed)
            continue
        paper = normalise_work(raw, hop=0)
        if paper is None:
            stats.seeds_missing.append(seed)
            continue
        _persist_paper(db, paper)
        seen.add(paper["paper_id"])
        stats.seeds_resolved.append(paper["paper_id"])
        queue.append((paper["paper_id"], 0))
        for reference in paper["references"][:max_refs_per_paper]:
            pending_citations.append((paper["paper_id"], reference))

    if not stats.seeds_resolved:
        log.error("no seeds resolved; corpus not built")
        stats.fetch = client.stats.to_dict()
        return stats

    while queue:
        if len(seen) >= max_papers:
            stats.frontier_truncated = True
            break

        paper_id, hop = queue.popleft()
        if hop >= hops:
            continue

        stored = db.get_paper(paper_id)
        if stored is None:
            continue

        # --- outgoing: the references recorded when this paper was persisted ---
        wanted = [
            reference
            for citing, reference in pending_citations
            if citing == paper_id and reference not in seen
        ][:max_refs_per_paper]

        fetched, failed_count = client.works_batch(wanted)
        if failed_count:
            stats.expansion_calls_failed += 1
        else:
            stats.papers_expanded += 1

        for raw in fetched:
            neighbour = normalise_work(raw, hop=hop + 1)
            if neighbour is None or neighbour["paper_id"] in seen:
                continue
            if len(seen) >= max_papers:
                stats.frontier_truncated = True
                break
            _persist_paper(db, neighbour)
            seen.add(neighbour["paper_id"])
            for reference in neighbour["references"][:max_refs_per_paper]:
                pending_citations.append((neighbour["paper_id"], reference))
            if hop + 1 < hops:
                queue.append((neighbour["paper_id"], hop + 1))

        # --- incoming: works that cite this one ---
        try:
            citing = client.citing_works(paper_id, limit=max_cites_per_paper)
        except FetchFailed as exc:
            log.warning("citing-works failed for %s: %s", paper_id, exc)
            stats.expansion_calls_failed += 1
            citing = []

        for raw in citing:
            neighbour = normalise_work(raw, hop=hop + 1)
            if neighbour is None:
                continue
            neighbour_id = neighbour["paper_id"]
            if neighbour_id not in seen:
                if len(seen) >= max_papers:
                    stats.frontier_truncated = True
                    continue
                _persist_paper(db, neighbour)
                seen.add(neighbour_id)
                if hop + 1 < hops:
                    queue.append((neighbour_id, hop + 1))
            for reference in neighbour["references"][:max_refs_per_paper]:
                pending_citations.append((neighbour_id, reference))

        if progress:
            log.info("corpus: %d papers, queue %d", len(seen), len(queue))

    # Citation edges are added only between papers that are both IN the corpus.
    # A reference to a paper we never fetched has no text and cannot be ranked,
    # so materialising it would add a phantom node that silently absorbs PPR mass.
    edge_rows = [
        (citing, "cites", cited, 1.0, {})
        for citing, cited in set(pending_citations)
        if citing in seen and cited in seen and citing != cited
    ]
    db.add_edges(edge_rows)

    stats.fetch = client.stats.to_dict()
    return _finalise(db, stats)


# -------------------------------------------------------------- concepts


class ConceptExtractor:
    """s2FieldsOfStudy + YAKE keyphrases over title+abstract.

    YAKE rather than KeyBERT (see config.yaml for the full justification): YAKE is
    statistical and deterministic, so a concept node's identity does not move when
    the dense retriever is swapped. KeyBERT would couple two components we need to
    vary independently.
    """

    def __init__(
        self,
        top_k: int = 8,
        ngram_max: int = 3,
        dedupe_threshold: float = 0.85,
        language: str = "en",
    ):
        self.top_k = top_k
        self.ngram_max = ngram_max
        self.dedupe_threshold = dedupe_threshold
        self.language = language
        self._extractor = None

    def _yake(self):
        if self._extractor is None:
            import yake  # imported lazily: corpus loading should not need it

            self._extractor = yake.KeywordExtractor(
                lan=self.language,
                n=self.ngram_max,
                dedupLim=self.dedupe_threshold,
                top=self.top_k,
                features=None,
            )
        return self._extractor

    def keyphrases(self, text: str) -> list[tuple[str, float]]:
        text = (text or "").strip()
        if len(text) < 20:
            return []
        try:
            raw = self._yake().extract_keywords(text)
        except Exception as exc:  # YAKE raises on some degenerate inputs
            log.debug("yake failed on a document: %s", exc)
            return []
        # YAKE scores are "lower is better"; invert to a weight in (0, 1] so edge
        # weight means the same direction as everywhere else in the system.
        results = []
        for phrase, score in raw:
            phrase = phrase.strip().lower()
            if len(phrase) < 3:
                continue
            results.append((phrase, 1.0 / (1.0 + float(score))))
        return results


def extract_concepts(
    db: Database,
    top_k: int = 8,
    ngram_max: int = 3,
    dedupe_threshold: float = 0.85,
    min_papers_per_concept: int = 2,
    progress: bool = True,
) -> dict[str, Any]:
    """Build concept nodes and `shares_concept` edges over the stored corpus.

    Two-pass: collect candidate concepts per paper, then keep only concepts that
    connect at least `min_papers_per_concept` papers. A concept touching a single
    paper cannot lie on a path between two papers, so it can never appear in an
    explanation and would only inflate the graph.
    """
    extractor = ConceptExtractor(top_k, ngram_max, dedupe_threshold)
    papers = db.all_papers()

    candidates: dict[str, list[tuple[str, float, str]]] = {}
    labels: dict[str, str] = {}

    for i, paper in enumerate(papers):
        pid = paper["paper_id"]

        for concept in paper.get("fields") or []:
            key = concept_node_id(concept)
            labels.setdefault(key, concept)
            candidates.setdefault(key, []).append((pid, 1.0, "s2_field"))

        text = f"{paper['title']}. {paper['abstract']}".strip()
        for phrase, weight in extractor.keyphrases(text):
            key = concept_node_id(phrase)
            if not key or key == "concept:":
                continue
            labels.setdefault(key, phrase)
            candidates.setdefault(key, []).append((pid, weight, "keyphrase"))

        if progress and i and i % 200 == 0:
            log.info("concepts: %d/%d papers", i, len(papers))

    kept = 0
    dropped = 0
    edge_rows: list[tuple[str, str, str, float, dict[str, Any]]] = []
    node_rows: list[tuple[str, str, str | None, dict[str, Any]]] = []

    for key, entries in candidates.items():
        distinct_papers = {pid for pid, _, _ in entries}
        if len(distinct_papers) < min_papers_per_concept:
            dropped += 1
            continue
        kept += 1
        node_rows.append((key, "concept", labels.get(key, key), {"degree": len(distinct_papers)}))
        # Collapse duplicate (paper, concept) pairs, keeping the strongest weight
        # and recording that the concept was reached both ways when it was.
        best: dict[str, tuple[float, set[str]]] = {}
        for pid, weight, origin in entries:
            current_weight, origins = best.get(pid, (0.0, set()))
            origins.add(origin)
            best[pid] = (max(current_weight, weight), origins)
        for pid, (weight, origins) in best.items():
            edge_rows.append(
                (pid, "shares_concept", key, weight, {"origin": sorted(origins)})
            )

    db.upsert_nodes(node_rows)
    db.add_edges(edge_rows)

    # `shares_concept` is one of the five relations the ablation experiment rests
    # on. A corpus that ends up with no concept nodes at all (e.g. an ingest
    # regression leaving abstracts empty) would make every shared_concept
    # explanation untestable, and in the results table that is indistinguishable
    # from "the concept signal is genuinely weak". Same guard shape as
    # `expansion_failed`.
    if papers and kept == 0:
        log.error(
            "NO CONCEPT NODES survived from %d papers (%d dropped as singletons). "
            "shares_concept explanations will all be empty -- check that abstracts "
            "were actually ingested.",
            len(papers), dropped,
        )

    return {
        "concepts_kept": kept,
        "concepts_dropped_singleton": dropped,
        "shares_concept_edges": len(edge_rows),
        "concepts_failed": bool(papers) and kept == 0,
    }


# ----------------------------------------------------------- corpus loaders


def load_s2orc(
    db: Database, path: str | Path, max_papers: int = 5000, source: str = "s2orc"
) -> CorpusStats:
    """Load an S2ORC subset (JSONL, one record per line).

    Expects the public S2ORC metadata shape: `corpusid`, `title`, `abstract`,
    `year`, `venue`, `authors`, and either `citations`/`references` id lists or an
    `outbound_citations` field. Unknown extra keys are ignored.
    """
    stats = CorpusStats()
    pending_edges: list[tuple[str, str]] = []
    loaded = 0

    for record in _iter_json_records(path):
        if loaded >= max_papers:
            stats.frontier_truncated = True
            break
        paper = _s2orc_record_to_paper(record, source)
        if paper is None:
            continue
        _persist_paper(db, paper)
        loaded += 1
        for cited in _citation_ids(record):
            pending_edges.append((paper["paper_id"], str(cited)))

    _add_internal_citation_edges(db, pending_edges)
    return _finalise(db, stats)


def load_dblp(
    db: Database, path: str | Path, max_papers: int = 5000, source: str = "dblp"
) -> CorpusStats:
    """Load DBLP-citation-v14 (JSON array or JSONL).

    Field names follow the ArnetMiner release: `id`, `title`, `abstract`,
    `year`, `venue` (object or string), `authors`, `references`, `n_citation`.
    """
    stats = CorpusStats()
    pending_edges: list[tuple[str, str]] = []
    loaded = 0

    for record in _iter_json_records(path):
        if loaded >= max_papers:
            stats.frontier_truncated = True
            break
        paper = _dblp_record_to_paper(record, source)
        if paper is None:
            continue
        _persist_paper(db, paper)
        loaded += 1
        for cited in record.get("references") or []:
            pending_edges.append((paper["paper_id"], str(cited)))

    _add_internal_citation_edges(db, pending_edges)
    return _finalise(db, stats)


def _iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read JSONL or a top-level JSON array, streaming where possible."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"corpus file not found: {file_path}. "
            "See docs/CORPORA.md for how to obtain S2ORC and DBLP-citation-v14."
        )

    with file_path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)

        if first == "[":
            # Whole-array load. DBLP v14 is large; documented as needing a
            # pre-sliced subset rather than the full dump.
            payload = json.load(handle)
            for record in payload:
                if isinstance(record, dict):
                    yield record
            return

        for line in handle:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _s2orc_record_to_paper(record: Mapping[str, Any], source: str) -> dict[str, Any] | None:
    raw_id = record.get("corpusid") or record.get("corpusId") or record.get("paper_id")
    if raw_id is None:
        return None
    abstract = record.get("abstract") or ""
    if isinstance(abstract, Mapping):
        abstract = abstract.get("text") or ""

    venue = record.get("venue") or ""
    if isinstance(venue, Mapping):
        venue = venue.get("name") or ""

    return {
        "paper_id": f"s2orc:{raw_id}",
        "corpus_id": str(raw_id),
        "title": (record.get("title") or "").strip(),
        "abstract": str(abstract).strip(),
        "year": record.get("year"),
        "venue": str(venue).strip(),
        "citation_count": record.get("citationcount") or record.get("n_citation") or 0,
        "reference_count": record.get("referencecount") or 0,
        "fields": _string_list(record.get("s2fieldsofstudy") or record.get("fields")),
        "authors": _authors(record.get("authors"), prefix="s2orc"),
        "source": source,
        "hop": None,
    }


def _dblp_record_to_paper(record: Mapping[str, Any], source: str) -> dict[str, Any] | None:
    raw_id = record.get("id") or record.get("_id")
    if raw_id is None:
        return None
    venue = record.get("venue") or ""
    if isinstance(venue, Mapping):
        venue = venue.get("raw") or venue.get("name") or ""

    return {
        "paper_id": f"dblp:{raw_id}",
        "corpus_id": str(raw_id),
        "title": (record.get("title") or "").strip(),
        "abstract": (record.get("abstract") or "").strip(),
        "year": record.get("year"),
        "venue": str(venue).strip(),
        "citation_count": record.get("n_citation") or 0,
        "reference_count": len(record.get("references") or []),
        "fields": _string_list(record.get("fos") or record.get("keywords")),
        "authors": _authors(record.get("authors"), prefix="dblp"),
        "source": source,
        "hop": None,
    }


def _citation_ids(record: Mapping[str, Any]) -> list[str]:
    """Outbound citation ids from an S2ORC record.

    The release has used several field names across versions
    (`outbound_citations`, `references`, `citations`), so all three are accepted
    and merged. Entries may be bare ids or objects carrying one.
    """
    ids: list[str] = []
    for key in ("outbound_citations", "outboundcitations", "references", "citations"):
        for entry in record.get(key) or []:
            if isinstance(entry, Mapping):
                value = entry.get("corpusid") or entry.get("corpusId") or entry.get("id")
            else:
                value = entry
            if value is not None:
                ids.append(str(value))
    # Preserve first-seen order while de-duplicating: the same reference can
    # appear under two of the accepted field names.
    return list(dict.fromkeys(ids))


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("category") or item.get("w")
            if name:
                result.append(str(name))
    return result


def _authors(value: Any, prefix: str) -> list[dict[str, str]]:
    if not value:
        return []
    authors = []
    for item in value:
        if isinstance(item, Mapping):
            author_id = item.get("authorId") or item.get("id") or item.get("name")
            name = item.get("name") or str(author_id)
        else:
            author_id = str(item)
            name = str(item)
        if author_id:
            authors.append({"author_id": f"{prefix}:{slugify(str(author_id))}", "name": name})
    return authors


def _add_internal_citation_edges(db: Database, pending: Sequence[tuple[str, str]]) -> None:
    """Add citation edges only where BOTH endpoints are in the corpus.

    Dangling citations to papers outside the loaded subset are dropped rather
    than creating phantom nodes with no text, which would be unrankable and would
    quietly distort PPR mass.
    """
    if not pending:
        return
    prefix = pending[0][0].split(":", 1)[0]
    known = {
        row["paper_id"] for row in db.all_papers() if row["paper_id"].startswith(f"{prefix}:")
    }
    rows = []
    for citing, cited_raw in pending:
        cited = cited_raw if ":" in cited_raw else f"{prefix}:{cited_raw}"
        if cited in known and citing in known and cited != citing:
            rows.append((citing, "cites", cited, 1.0, {}))
    db.add_edges(rows)
