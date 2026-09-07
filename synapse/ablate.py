"""Phase 4 — THE CONTRIBUTION: counterfactual edge ablation.

The question: when a retrieval system states why it ranked a paper where it did,
did that stated reason actually cause the ranking?

Protocol, per (seed, candidate, explainer):
  1. baseline rank r0
  2. delete exactly the edges the explanation cited
  3. recompute PPR and re-rank with IDENTICAL parameters
  4. ablated rank r1

The random-edge control is built first and defined at the top of this file,
because it is the thing that stops the whole result being dismissed. Deleting
edges moves ranks. The only interesting question is whether deleting the CITED
edges moves them more than deleting the same number of arbitrary ones.

A candidate is called faithful only if the observed displacement beats its own
control distribution at the configured alpha. Everything else is reported as
unfaithful, including the cases where the explanation cited nothing at all.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .explain import Explainer, Explanation
from .graph import TypedGraph
from .metrics import (
    Distribution,
    PairedTest,
    bootstrap_distribution,
    min_achievable_p_value,
    monte_carlo_p_value,
    paired_permutation_test,
)
from .retrieve import Retriever

log = logging.getLogger(__name__)


# ------------------------------------------------------- the control, first


def sample_control_edges(
    graph: TypedGraph,
    seed_paper: str,
    candidate: str,
    count: int,
    exclude: Iterable[str],
    rng: random.Random,
    disjoint: bool = False,
) -> list[str] | None:
    """Draw `count` random edges from the same structural pool the explainer used.

    Pool = edges incident to the seed or the candidate. That is where every
    explanation's edges come from, so drawing from anywhere else would compare
    the explanation against a strictly weaker intervention and manufacture
    faithfulness.

    THE CITED EDGES STAY IN THE POOL (`disjoint=False`, the default).
    The null hypothesis being tested is exchangeability: "these edges are no more
    special than any other n incident edges". The correct null distribution for
    that is a uniform draw from the whole pool, exactly as in a permutation test.

    Forcing the control to avoid the cited edges was the first implementation and
    was wrong twice over. It tests a different, weaker null ("some OTHER n edges"),
    and it becomes infeasible precisely when an explanation cites most of the
    neighbourhood -- so the cases it silently dropped were the broad explanations,
    which is a bias toward whichever explainer cites least. Leaving them in also
    makes the test strictly harder to pass, which is the right direction for a
    protocol whose headline result may be a null finding.

    Returns None when the pool is smaller than `count` -- the caller records an
    untestable case rather than comparing unequally sized interventions.
    """
    pool = set(graph.incident_edge_ids(seed_paper)) | set(
        graph.incident_edge_ids(candidate)
    )
    if disjoint:
        pool -= set(exclude)
    if len(pool) < count:
        return None
    ordered = sorted(pool)  # deterministic ordering before a seeded draw
    return rng.sample(ordered, count)


# ------------------------------------------------------------------ results


@dataclass
class CandidateAblation:
    """One (seed, candidate, explainer) cell of the experiment."""

    seed: str
    candidate: str
    explainer: str

    baseline_rank: int
    ablated_rank: int
    displacement: int
    necessity_score: float

    control_displacements: list[int] = field(default_factory=list)
    control_mean: float = 0.0
    control_necessity_mean: float = 0.0
    p_value: float = 1.0
    faithful: bool = False

    sufficiency_rank: int | None = None
    sufficiency_retention: float | None = None

    cited_edge_count: int = 0
    cited_edges: list[str] = field(default_factory=list)
    explanation_text: str = ""
    testable: bool = True
    untestable_reason: str | None = None
    unrealised_claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "candidate": self.candidate,
            "explainer": self.explainer,
            "baseline_rank": self.baseline_rank,
            "ablated_rank": self.ablated_rank,
            "displacement": self.displacement,
            "necessity_score": self.necessity_score,
            "control_displacements": self.control_displacements,
            "control_mean": self.control_mean,
            "control_necessity_mean": self.control_necessity_mean,
            "p_value": self.p_value,
            "faithful": self.faithful,
            "sufficiency_rank": self.sufficiency_rank,
            "sufficiency_retention": self.sufficiency_retention,
            "cited_edge_count": self.cited_edge_count,
            "cited_edges": self.cited_edges,
            "explanation_text": self.explanation_text,
            "testable": self.testable,
            "untestable_reason": self.untestable_reason,
            "unrealised_claims": self.unrealised_claims,
        }


@dataclass
class ExplainerReport:
    """Aggregate over all candidates for one explainer. Never a bare mean."""

    explainer: str
    n_candidates: int
    n_testable: int
    n_faithful: int
    displacement: Distribution
    necessity: Distribution
    control_displacement: Distribution
    control_necessity: Distribution
    paired_test: PairedTest
    sufficiency_retention: Distribution
    mean_cited_edges: float
    empty_explanations: int
    unrealised_claim_count: int

    @property
    def faithful_fraction(self) -> float:
        return self.n_faithful / self.n_testable if self.n_testable else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "explainer": self.explainer,
            "n_candidates": self.n_candidates,
            "n_testable": self.n_testable,
            "n_faithful": self.n_faithful,
            "faithful_fraction": self.faithful_fraction,
            "displacement": self.displacement.to_dict(),
            "necessity": self.necessity.to_dict(),
            "control_displacement": self.control_displacement.to_dict(),
            "control_necessity": self.control_necessity.to_dict(),
            "paired_test": self.paired_test.to_dict(),
            "sufficiency_retention": self.sufficiency_retention.to_dict(),
            "mean_cited_edges": self.mean_cited_edges,
            "empty_explanations": self.empty_explanations,
            "unrealised_claim_count": self.unrealised_claim_count,
        }


# ------------------------------------------------------------------ scoring


def necessity_score(baseline_rank: int, ablated_rank: int, list_length: int) -> float:
    """Displacement normalised by the room the candidate had to fall.

    A candidate at rank 2 in a 500-paper ranking can fall 498 places; one at
    rank 480 can fall 20. Raw displacement therefore is not comparable across
    candidates, and averaging it would systematically favour explanations
    attached to top-ranked papers.

        necessity = (r1 - r0) / (N - r0)

    1.0 means the candidate fell as far as it possibly could. 0.0 means it did
    not move. Negative means the explanation's edges were holding the candidate
    DOWN, which is a real and reportable outcome, so it is not clipped.
    """
    room = list_length - baseline_rank
    if room <= 0:
        return 0.0
    return (ablated_rank - baseline_rank) / room


# ------------------------------------------------------------------ harness


class AblationHarness:
    def __init__(
        self,
        retriever: Retriever,
        config,
        seed: int | None = None,
    ):
        self.retriever = retriever
        self.config = config
        self.rng_seed = int(seed if seed is not None else config.require("seed"))

        self.control_trials = int(config.get_path("ablate.random_control_trials", 30))
        self.bootstrap_samples = int(config.get_path("ablate.bootstrap_samples", 2000))
        self.confidence = float(config.get_path("ablate.confidence_level", 0.95))
        self.alpha = float(config.get_path("ablate.significance_alpha", 0.05))
        self.do_sufficiency = bool(config.get_path("ablate.sufficiency", True))

        floor = min_achievable_p_value(self.control_trials)
        if floor >= self.alpha:
            raise ValueError(
                f"random_control_trials={self.control_trials} gives a minimum "
                f"p-value of {floor:.3f}, which is not below alpha={self.alpha}. "
                "No candidate could ever be called faithful. Raise the trial count."
            )

        self._rank_cache: dict[tuple[str, frozenset[str]], dict[str, int]] = {}

    # -------------------------------------------------------------- ranking

    def _full_ranking(self, query: str, graph: TypedGraph) -> dict[str, int]:
        """Rank the whole corpus under a given graph. Memoised per edge set.

        Full-corpus rather than top-k: a candidate pushed out of the top 20 must
        get its true new rank, not a sentinel. A sentinel would make every strong
        displacement look identical and destroy the metric's resolution.
        """
        key = (query, graph.edge_ids)
        cached = self._rank_cache.get(key)
        if cached is not None:
            return cached

        result = self.retriever.search(
            query,
            top_k=len(self.retriever.paper_ids),
            graph=graph,
            signals="rrf+ppr",
            pool_size=len(self.retriever.paper_ids),
            attribute=False,
        )
        ranks = result.ranks()
        self._rank_cache[key] = ranks
        return ranks

    def _rank_of(self, ranks: Mapping[str, int], candidate: str) -> int:
        """Rank, or one past the end when the candidate is unranked."""
        return ranks.get(candidate, len(self.retriever.paper_ids) + 1)

    # ------------------------------------------------------------- one cell

    def ablate_candidate(
        self,
        seed: str,
        candidate: str,
        explanation: Explanation,
        baseline_ranks: Mapping[str, int],
    ) -> CandidateAblation:
        graph = self.retriever.graph
        list_length = len(self.retriever.paper_ids)
        baseline_rank = self._rank_of(baseline_ranks, candidate)

        cited = [e for e in explanation.cited_edges if e in graph.edge_ids]

        if not cited:
            # An explanation that cites nothing cannot be tested for necessity.
            # It is counted, reported, and excluded from the significance test --
            # scoring it as displacement 0 would let empty explanations drag an
            # explainer's mean toward "harmless".
            return CandidateAblation(
                seed=seed,
                candidate=candidate,
                explainer=explanation.explainer,
                baseline_rank=baseline_rank,
                ablated_rank=baseline_rank,
                displacement=0,
                necessity_score=0.0,
                cited_edge_count=0,
                cited_edges=[],
                explanation_text=explanation.text,
                testable=False,
                untestable_reason="explanation cited no edges present in the graph",
                unrealised_claims=list(explanation.unrealised_claims),
            )

        ablated_graph = graph.without_edges(cited)
        ablated_ranks = self._full_ranking(seed, ablated_graph)
        ablated_rank = self._rank_of(ablated_ranks, candidate)
        displacement = ablated_rank - baseline_rank
        necessity = necessity_score(baseline_rank, ablated_rank, list_length)

        # ---- the control, run for every single cell, never optional ----
        rng = random.Random(f"{self.rng_seed}:{seed}:{candidate}:{explanation.explainer}")
        control_displacements: list[int] = []
        control_necessities: list[float] = []
        control_failed = False

        for _ in range(self.control_trials):
            control_edges = sample_control_edges(
                graph, seed, candidate, len(cited), exclude=cited, rng=rng
            )
            if control_edges is None:
                control_failed = True
                break
            control_ranks = self._full_ranking(seed, graph.without_edges(control_edges))
            control_rank = self._rank_of(control_ranks, candidate)
            control_displacements.append(control_rank - baseline_rank)
            control_necessities.append(
                necessity_score(baseline_rank, control_rank, list_length)
            )

        if control_failed:
            return CandidateAblation(
                seed=seed,
                candidate=candidate,
                explainer=explanation.explainer,
                baseline_rank=baseline_rank,
                ablated_rank=ablated_rank,
                displacement=displacement,
                necessity_score=necessity,
                cited_edge_count=len(cited),
                cited_edges=cited,
                explanation_text=explanation.text,
                testable=False,
                untestable_reason=(
                    f"too few incident edges to draw a {len(cited)}-edge control"
                ),
                unrealised_claims=list(explanation.unrealised_claims),
            )

        p_value = monte_carlo_p_value(float(displacement), control_displacements)

        sufficiency_rank = None
        sufficiency_retention = None
        if self.do_sufficiency:
            kept_graph = graph.only_edges(cited)
            sufficiency_ranks = self._full_ranking(seed, kept_graph)
            sufficiency_rank = self._rank_of(sufficiency_ranks, candidate)
            # 1.0 = the cited edges alone reproduce the original rank exactly.
            room = list_length - baseline_rank
            sufficiency_retention = (
                1.0 - ((sufficiency_rank - baseline_rank) / room) if room > 0 else 1.0
            )

        return CandidateAblation(
            seed=seed,
            candidate=candidate,
            explainer=explanation.explainer,
            baseline_rank=baseline_rank,
            ablated_rank=ablated_rank,
            displacement=displacement,
            necessity_score=necessity,
            control_displacements=control_displacements,
            control_mean=float(np.mean(control_displacements)),
            control_necessity_mean=float(np.mean(control_necessities)),
            p_value=p_value,
            faithful=p_value < self.alpha and displacement > 0,
            sufficiency_rank=sufficiency_rank,
            sufficiency_retention=sufficiency_retention,
            cited_edge_count=len(cited),
            cited_edges=cited,
            explanation_text=explanation.text,
            testable=True,
            unrealised_claims=list(explanation.unrealised_claims),
        )

    # ------------------------------------------------------------ full sweep

    def run(
        self,
        seed: str,
        explainers: Mapping[str, Explainer],
        top_k: int = 20,
        progress: bool = True,
    ) -> tuple[list[CandidateAblation], dict[str, ExplainerReport], dict[str, Any]]:
        """Ablate every (candidate, explainer) pair for one seed paper."""
        baseline = self.retriever.search(seed, top_k=top_k, signals="rrf+ppr")
        baseline_ranks = self._full_ranking(seed, self.retriever.graph)

        candidates = [c.paper_id for c in baseline.candidates]
        results: list[CandidateAblation] = []

        for explainer_name, explainer in explainers.items():
            for i, candidate in enumerate(candidates, start=1):
                explanation = explainer.explain(seed, candidate)
                results.append(
                    self.ablate_candidate(seed, candidate, explanation, baseline_ranks)
                )
                if progress:
                    log.info("[%s] %d/%d %s", explainer_name, i, len(candidates), candidate)

        reports = {
            name: self.summarise(name, [r for r in results if r.explainer == name])
            for name in explainers
        }

        diagnostics = {
            "seed": seed,
            "list_length": len(self.retriever.paper_ids),
            "top_k": top_k,
            "control_trials": self.control_trials,
            "alpha": self.alpha,
            "min_achievable_p": min_achievable_p_value(self.control_trials),
            "baseline": baseline.to_dict(),
            "unique_graphs_evaluated": len(self._rank_cache),
        }
        return results, reports, diagnostics

    def summarise(self, explainer: str, cells: Sequence[CandidateAblation]) -> ExplainerReport:
        testable = [c for c in cells if c.testable]

        displacements = [float(c.displacement) for c in testable]
        necessities = [c.necessity_score for c in testable]
        control_means = [c.control_mean for c in testable]
        control_necessity_means = [c.control_necessity_mean for c in testable]
        retentions = [
            c.sufficiency_retention for c in testable if c.sufficiency_retention is not None
        ]

        return ExplainerReport(
            explainer=explainer,
            n_candidates=len(cells),
            n_testable=len(testable),
            n_faithful=sum(1 for c in testable if c.faithful),
            displacement=bootstrap_distribution(
                displacements, self.bootstrap_samples, self.confidence, self.rng_seed
            ),
            necessity=bootstrap_distribution(
                necessities, self.bootstrap_samples, self.confidence, self.rng_seed
            ),
            control_displacement=bootstrap_distribution(
                control_means, self.bootstrap_samples, self.confidence, self.rng_seed
            ),
            control_necessity=bootstrap_distribution(
                control_necessity_means, self.bootstrap_samples, self.confidence, self.rng_seed
            ),
            paired_test=paired_permutation_test(
                displacements, control_means, alpha=self.alpha, seed=self.rng_seed
            ),
            sufficiency_retention=bootstrap_distribution(
                retentions, self.bootstrap_samples, self.confidence, self.rng_seed
            ),
            mean_cited_edges=(
                float(np.mean([c.cited_edge_count for c in cells])) if cells else 0.0
            ),
            empty_explanations=sum(1 for c in cells if c.cited_edge_count == 0),
            unrealised_claim_count=sum(len(c.unrealised_claims) for c in cells),
        )
