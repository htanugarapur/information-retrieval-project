"""Phase 2 — retrieval, where every candidate carries its own provenance.

Four signals: BM25 (sparse), dense embeddings, Reciprocal Rank Fusion over the
two, and Personalized PageRank over the typed graph. The final score is a
weighted sum of RRF and PPR.

The point of this module is not the ranking. It is that every candidate can say
exactly why it is where it is: which sparse rank, which dense rank, which PPR
mass, and — the field the whole paper depends on — which graph edges delivered
that mass, recorded during propagation by `synapse.ppr`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Config
from .db import Database
from .graph import TypedGraph
from .ppr import EdgeContribution, FlowIndex, PPRResult, attribute_edges, personalized_pagerank

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately trivial: a fancier analyser
    would be one more thing to hold fixed when isolating causes."""
    return _TOKEN.findall((text or "").lower())


@dataclass
class Candidate:
    """One ranked paper with a complete attribution record."""

    paper_id: str
    final_score: float
    rank: int = 0

    bm25_rank: int | None = None
    bm25_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rrf_norm: float = 0.0
    ppr_score: float = 0.0
    ppr_norm: float = 0.0

    contributing_edges: list[tuple[str, str, str]] = field(default_factory=list)
    edge_contributions: list[EdgeContribution] = field(default_factory=list)

    title: str = ""
    year: int | None = None
    venue: str = ""
    citation_count: int = 0

    def cited_edge_ids(self) -> list[str]:
        return [c.edge_id for c in self.edge_contributions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "rank": self.rank,
            "final_score": self.final_score,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "rrf_score": self.rrf_score,
            "rrf_norm": self.rrf_norm,
            "ppr_score": self.ppr_score,
            "ppr_norm": self.ppr_norm,
            "contributing_edges": [list(t) for t in self.contributing_edges],
            "edge_contributions": [c.to_dict() for c in self.edge_contributions],
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
        }


@dataclass
class RetrievalResult:
    query: str
    seed_paper_id: str | None
    candidates: list[Candidate]
    mode: str
    ppr_diagnostics: dict[str, Any] = field(default_factory=dict)

    def ranks(self) -> dict[str, int]:
        return {c.paper_id: c.rank for c in self.candidates}

    def rank_of(self, paper_id: str) -> int | None:
        for candidate in self.candidates:
            if candidate.paper_id == paper_id:
                return candidate.rank
        return None

    def get(self, paper_id: str) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.paper_id == paper_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "seed_paper_id": self.seed_paper_id,
            "mode": self.mode,
            "ppr_diagnostics": self.ppr_diagnostics,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ------------------------------------------------------------------ indexes


class SparseIndex:
    """BM25 over title + abstract."""

    def __init__(self, paper_ids: Sequence[str], documents: Sequence[str], k1: float, b: float):
        from rank_bm25 import BM25Okapi

        self.paper_ids = list(paper_ids)
        self.index = {pid: i for i, pid in enumerate(self.paper_ids)}
        self._bm25 = BM25Okapi([tokenize(doc) for doc in documents], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.paper_ids), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)


class DenseIndex:
    """sentence-transformers embeddings with cosine similarity.

    Embeddings are computed once per corpus and cached in memory. The model is
    config-swappable (MiniLM by default, SPECTER2 for the sensitivity check).
    """

    def __init__(
        self,
        paper_ids: Sequence[str],
        documents: Sequence[str],
        model_name: str,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.paper_ids = list(paper_ids)
        self.index = {pid: i for i, pid in enumerate(self.paper_ids)}
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self._embeddings = self._model.encode(
            list(documents),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float64)
        self._normalize = normalize

    def scores(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        ).astype(np.float64)[0]
        return self._embeddings @ vector


# ------------------------------------------------------------------ fusion


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """1-based ranks, highest score first. Ties broken by index for determinism."""
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    return ranks


def reciprocal_rank_fusion(rank_lists: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    """Standard RRF: sum of 1/(k + rank) across the input rankings."""
    total = np.zeros(rank_lists[0].shape[0], dtype=np.float64)
    for ranks in rank_lists:
        total += 1.0 / (k + ranks)
    return total


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] over the candidate pool.

    RRF scores live around 1/60 and PPR mass around 1/N; summing them raw would
    make the configured 0.6/0.4 weighting a lie. Normalising over the pool makes
    the weights mean what they say. Constant input maps to zeros, not NaN.
    """
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-15:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


# ------------------------------------------------------------------ retriever


class Retriever:
    """Holds the corpus indexes and answers queries with full provenance."""

    def __init__(self, db: Database, config: Config, load_dense: bool = True):
        self.db = db
        self.config = config

        self.papers = db.all_papers()
        self.paper_ids = [p["paper_id"] for p in self.papers]
        self.paper_by_id = {p["paper_id"]: p for p in self.papers}
        documents = [f"{p['title']}. {p['abstract']}".strip() for p in self.papers]

        self.sparse = SparseIndex(
            self.paper_ids,
            documents,
            k1=float(config.get_path("retrieval.bm25.k1", 1.5)),
            b=float(config.get_path("retrieval.bm25.b", 0.75)),
        )

        self.dense: DenseIndex | None = None
        if load_dense and self.paper_ids:
            try:
                self.dense = DenseIndex(
                    self.paper_ids,
                    documents,
                    model_name=str(config.get_path("retrieval.dense.model")),
                    batch_size=int(config.get_path("retrieval.dense.batch_size", 32)),
                    normalize=bool(config.get_path("retrieval.dense.normalize", True)),
                )
            except Exception as exc:
                # A missing model must be loud. Silently falling back to
                # sparse-only would change every number in the paper without
                # changing the config that supposedly describes the run.
                raise RuntimeError(
                    f"dense index failed to load ({exc}). "
                    "Pass load_dense=False to run sparse-only deliberately."
                ) from exc

        self.graph = TypedGraph.from_db(
            db, edge_weights=config.get_path("retrieval.ppr.edge_weights") or {}
        )
        self._flow_index = FlowIndex(self.graph)

        # BM25 and dense scores depend only on the query text, never on the
        # graph. The ablation harness re-ranks the same query dozens of times per
        # candidate against counterfactual graphs, so memoising here turns an
        # O(ablations) embedding cost into O(1). Correctness relies on the
        # invariant that no ablation touches paper text -- asserted in tests.
        self._lexical_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ query

    def search(
        self,
        query: str,
        top_k: int | None = None,
        graph: TypedGraph | None = None,
        signals: str = "rrf+ppr",
        pool_size: int | None = None,
        attribute: bool = True,
    ) -> RetrievalResult:
        """Rank the corpus against a query string or a paper id.

        `graph` overrides the corpus graph — that is how the ablation harness
        re-ranks against a counterfactual graph using identical parameters.
        `signals` selects the retrieval ablation: bm25-only, dense-only,
        rrf-only, or rrf+ppr.
        """
        top_k = top_k or int(self.config.get_path("retrieval.top_k", 20))
        pool_size = pool_size or int(self.config.get_path("retrieval.candidate_pool", 200))
        active_graph = graph if graph is not None else self.graph

        if not self.paper_ids:
            return RetrievalResult(query, None, [], signals)

        seed_paper_id = query if query in self.paper_by_id else None
        if seed_paper_id:
            paper = self.paper_by_id[seed_paper_id]
            query_text = f"{paper['title']}. {paper['abstract']}".strip()
        else:
            query_text = query

        bm25_scores, dense_scores = self._lexical_scores(query_text)
        bm25_ranks = rank_positions(bm25_scores)
        dense_ranks = rank_positions(dense_scores)

        # One k for all three branches. The single-signal branches previously
        # hardcoded 60 while the fusion branch read config, so changing
        # retrieval.rrf.k silently scored rows of the SAME eval table under
        # different constants.
        rrf_k = int(self.config.get_path("retrieval.rrf.k", 60))
        if signals == "bm25-only":
            rrf_scores = 1.0 / (rrf_k + bm25_ranks)
        elif signals == "dense-only":
            rrf_scores = 1.0 / (rrf_k + dense_ranks)
        else:
            rrf_scores = reciprocal_rank_fusion([bm25_ranks, dense_ranks], k=rrf_k)

        # Exclude the seed itself: a paper is not a retrieval result for itself,
        # and leaving it in would sit at rank 1 and shift every displacement.
        excluded = {seed_paper_id} if seed_paper_id else set()

        ppr_result: PPRResult | None = None
        ppr_scores = np.zeros_like(bm25_scores)
        diagnostics: dict[str, Any] = {"used": False}

        if signals == "rrf+ppr":
            seed_nodes = self._resolve_ppr_seeds(seed_paper_id, rrf_scores)
            ppr_result = personalized_pagerank(
                active_graph,
                seed_nodes,
                alpha=float(self.config.get_path("retrieval.ppr.alpha", 0.15)),
                max_iterations=int(self.config.get_path("retrieval.ppr.max_iterations", 60)),
                tolerance=float(self.config.get_path("retrieval.ppr.tolerance", 1e-10)),
            )
            ppr_scores = np.array(
                [ppr_result.score_of(active_graph, pid) for pid in self.paper_ids],
                dtype=np.float64,
            )
            diagnostics = {
                "used": True,
                "converged": ppr_result.converged,
                "iterations": ppr_result.iterations,
                "residual": ppr_result.residual,
                "seed_nodes": list(seed_nodes),
                # The number that certifies contributing_edges is exact, not
                # approximate. A reviewer can read it straight off the artifact.
                "decomposition_error": ppr_result.check_decomposition(active_graph),
            }

        # Build the pool from the strongest signals, then score within it.
        pool_rank_source = rrf_scores if signals != "rrf+ppr" else (
            min_max_normalize(rrf_scores) * float(self.config.get_path("retrieval.fusion.rrf_weight", 0.6))
            + min_max_normalize(ppr_scores) * float(self.config.get_path("retrieval.fusion.ppr_weight", 0.4))
        )
        candidate_indices = [
            i for i in np.argsort(-pool_rank_source, kind="stable")
            if self.paper_ids[i] not in excluded
        ][:pool_size]

        if not candidate_indices:
            return RetrievalResult(query, seed_paper_id, [], signals, diagnostics)

        pool = np.asarray(candidate_indices, dtype=np.int64)
        rrf_norm = min_max_normalize(rrf_scores[pool])
        ppr_norm = min_max_normalize(ppr_scores[pool])

        if signals == "rrf+ppr":
            rrf_weight = float(self.config.get_path("retrieval.fusion.rrf_weight", 0.6))
            ppr_weight = float(self.config.get_path("retrieval.fusion.ppr_weight", 0.4))
            final = rrf_weight * rrf_norm + ppr_weight * ppr_norm
        else:
            final = rrf_norm

        order = np.lexsort((pool, -final))

        attribution_cfg = self.config.get_path("retrieval.ppr.attribution") or {}
        flow_index = self._flow_index if graph is None else FlowIndex(active_graph)

        candidates: list[Candidate] = []
        for rank, position in enumerate(order[:top_k], start=1):
            idx = int(pool[position])
            paper_id = self.paper_ids[idx]
            paper = self.paper_by_id[paper_id]

            contributions: list[EdgeContribution] = []
            if attribute and ppr_result is not None:
                contributions = attribute_edges(
                    active_graph,
                    ppr_result,
                    paper_id,
                    max_depth=int(attribution_cfg.get("max_depth", 3)),
                    min_flow_fraction=float(attribution_cfg.get("min_flow_fraction", 0.01)),
                    max_edges=int(attribution_cfg.get("max_edges_per_candidate", 40)),
                    flow_index=flow_index,
                )

            candidates.append(
                Candidate(
                    paper_id=paper_id,
                    rank=rank,
                    final_score=float(final[position]),
                    bm25_rank=int(bm25_ranks[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    dense_rank=int(dense_ranks[idx]),
                    dense_score=float(dense_scores[idx]),
                    rrf_score=float(rrf_scores[idx]),
                    rrf_norm=float(rrf_norm[position]),
                    ppr_score=float(ppr_scores[idx]),
                    ppr_norm=float(ppr_norm[position]),
                    contributing_edges=[c.as_triple() for c in contributions],
                    edge_contributions=contributions,
                    title=paper["title"],
                    year=paper["year"],
                    venue=paper["venue"],
                    citation_count=paper["citation_count"],
                )
            )

        return RetrievalResult(query, seed_paper_id, candidates, signals, diagnostics)

    def _lexical_scores(self, query_text: str) -> tuple[np.ndarray, np.ndarray]:
        """Memoised (bm25, dense) score vectors for a query string."""
        cached = self._lexical_cache.get(query_text)
        if cached is not None:
            return cached
        bm25_scores = self.sparse.scores(query_text)
        if self.dense is not None:
            dense_scores = self.dense.scores(query_text)
        else:
            dense_scores = np.zeros_like(bm25_scores)
        # Stored read-only: a caller mutating these would corrupt every later
        # ablation of the same query, and the bug would look like a real result.
        bm25_scores.flags.writeable = False
        dense_scores.flags.writeable = False
        self._lexical_cache[query_text] = (bm25_scores, dense_scores)
        return bm25_scores, dense_scores

    def _resolve_ppr_seeds(
        self, seed_paper_id: str | None, rrf_scores: np.ndarray, top_n: int = 5
    ) -> dict[str, float]:
        """Personalization vector for PPR.

        A paper-id query seeds on that paper alone — the cleanest causal setup,
        since every path traced back leads to one known origin. A free-text query
        has no graph anchor, so it seeds on the top few lexical hits, and that
        weaker anchoring is recorded in the artifact rather than hidden.
        """
        if seed_paper_id:
            return {seed_paper_id: 1.0}
        top = np.argsort(-rrf_scores, kind="stable")[:top_n]
        return {self.paper_ids[int(i)]: float(rrf_scores[int(i)]) for i in top if rrf_scores[int(i)] > 0}

    # ----------------------------------------------------------------- helper

    def rerank_on_graph(self, result: RetrievalResult, graph: TypedGraph) -> RetrievalResult:
        """Re-run the identical query against a counterfactual graph."""
        return self.search(
            result.seed_paper_id or result.query,
            top_k=len(self.paper_ids),
            graph=graph,
            signals=result.mode,
            attribute=False,
        )
