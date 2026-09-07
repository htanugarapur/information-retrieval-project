"""Phase 3 — three explainers. The comparison between them IS the experiment.

All three implement `explain(seed, candidate) -> Explanation(text, cited_edges)`.
They differ only in what they are allowed to see:

  MetaPathExplainer     the graph, and nothing else. Deterministic traversal.
  LLMGroundedExplainer  the extracted meta-path FACTS, and nothing else.
  LLMFreeExplainer      titles and abstracts, and NOTHING about the graph.

METHODOLOGICAL NOTE ON THE FREE EXPLAINER (a reviewer will ask about this)
-------------------------------------------------------------------------
`LLMFreeExplainer` writes prose with no graph access. But an ablation can only
delete edges, so its prose must be mapped onto edges to be testable. That mapping
happens strictly AFTERWARDS, in `_ground_claims`, and the model never sees its
result. So the model's *reasoning* is graph-free, while its *claims* are scored
against the graph — which is the whole question being asked: when a system
free-associates a justification, do the things it asserts turn out to be the
things that actually moved the ranking?

Claims that correspond to no real edge are kept and counted
(`unrealised_claims`). They are the most interesting output of this explainer:
an explanation asserting a relationship the graph does not contain is
unfaithful in the strongest possible sense, and dropping those cases would
flatter the method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .db import Database
from .graph import TypedGraph
from .llm import LLMUnavailable, OpenRouterClient

log = logging.getLogger(__name__)

# The relation vocabulary shared by all three explainers. Fixed and closed: a
# free-text explainer inventing a new relation name must not silently create a
# new category, it must land in `unrealised_claims`.
RELATION_TYPES = (
    "direct_citation",
    "co_citation",
    "shared_author",
    "shared_concept",
    "shared_venue",
)


@dataclass(frozen=True)
class MetaPath:
    """One typed connection between seed and candidate, with its exact edges."""

    relation: str
    detail: str
    edge_ids: tuple[str, ...]
    via: str | None = None          # intermediate node id, when there is one
    via_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "detail": self.detail,
            "edge_ids": list(self.edge_ids),
            "via": self.via,
            "via_label": self.via_label,
        }


@dataclass
class Explanation:
    """What an explainer asserts, and exactly which edges it staked that on."""

    explainer: str
    seed: str
    candidate: str
    text: str
    cited_edges: list[str] = field(default_factory=list)
    paths: list[MetaPath] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    unrealised_claims: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.cited_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "explainer": self.explainer,
            "seed": self.seed,
            "candidate": self.candidate,
            "text": self.text,
            "cited_edges": list(self.cited_edges),
            "paths": [p.to_dict() for p in self.paths],
            "claims": self.claims,
            "unrealised_claims": self.unrealised_claims,
            "diagnostics": self.diagnostics,
        }


# --------------------------------------------------------------- meta-paths


class MetaPathFinder:
    """Deterministic typed-path enumeration between two papers.

    Shared by all three explainers: the metapath explainer emits these directly,
    the grounded explainer is shown them as facts, and the free explainer's
    claims are checked against them. One implementation means the three are
    compared on identical ground truth.
    """

    def __init__(self, graph: TypedGraph, db: Database, max_paths: int = 12):
        self.graph = graph
        self.db = db
        self.max_paths = max_paths
        self._label_cache: dict[str, str] = {}

    def label(self, node_id: str) -> str:
        if node_id not in self._label_cache:
            node = self.db.get_node(node_id)
            self._label_cache[node_id] = (node or {}).get("label") or node_id
        return self._label_cache[node_id]

    def _typed_neighbours(self, node_id: str, relation: str) -> dict[str, str]:
        """{neighbour_id: edge_id} for one directed relation."""
        return {
            neighbour: edge_id
            for rel, neighbour, edge_id in self.graph.neighbours(node_id, [relation])
        }

    def find(self, seed: str, candidate: str) -> list[MetaPath]:
        if not self.graph.has_node(seed) or not self.graph.has_node(candidate):
            return []

        paths: list[MetaPath] = []
        paths.extend(self._direct_citation(seed, candidate))
        paths.extend(self._co_citation(seed, candidate))
        paths.extend(self._shared_via(seed, candidate, "authored_by", "shared_author"))
        paths.extend(self._shared_via(seed, candidate, "shares_concept", "shared_concept"))
        paths.extend(self._shared_via(seed, candidate, "published_in", "shared_venue"))

        # Deterministic ordering, strongest evidence first. Direct citation
        # outranks a shared venue because a venue links thousands of papers.
        priority = {rel: i for i, rel in enumerate(RELATION_TYPES)}
        paths.sort(key=lambda p: (priority.get(p.relation, 99), p.via or "", p.detail))
        return paths[: self.max_paths]

    def _direct_citation(self, seed: str, candidate: str) -> list[MetaPath]:
        result = []
        forward = self._typed_neighbours(seed, "cites")
        if candidate in forward:
            result.append(
                MetaPath(
                    "direct_citation",
                    f"{self.label(seed)} cites {self.label(candidate)}",
                    (forward[candidate],),
                )
            )
        backward = self._typed_neighbours(candidate, "cites")
        if seed in backward:
            result.append(
                MetaPath(
                    "direct_citation",
                    f"{self.label(candidate)} cites {self.label(seed)}",
                    (backward[seed],),
                )
            )
        return result

    def _co_citation(self, seed: str, candidate: str) -> list[MetaPath]:
        """Papers both cite (bibliographic coupling) or that both cite them."""
        result = []

        seed_refs = self._typed_neighbours(seed, "cites")
        candidate_refs = self._typed_neighbours(candidate, "cites")
        for shared in sorted(set(seed_refs) & set(candidate_refs)):
            result.append(
                MetaPath(
                    "co_citation",
                    f"both cite {self.label(shared)}",
                    (seed_refs[shared], candidate_refs[shared]),
                    via=shared,
                    via_label=self.label(shared),
                )
            )

        seed_citers = self._typed_neighbours(seed, "cited_by")
        candidate_citers = self._typed_neighbours(candidate, "cited_by")
        for shared in sorted(set(seed_citers) & set(candidate_citers)):
            result.append(
                MetaPath(
                    "co_citation",
                    f"both cited by {self.label(shared)}",
                    (seed_citers[shared], candidate_citers[shared]),
                    via=shared,
                    via_label=self.label(shared),
                )
            )
        return result

    def _shared_via(
        self, seed: str, candidate: str, relation: str, label: str
    ) -> list[MetaPath]:
        seed_side = self._typed_neighbours(seed, relation)
        candidate_side = self._typed_neighbours(candidate, relation)
        result = []
        for shared in sorted(set(seed_side) & set(candidate_side)):
            result.append(
                MetaPath(
                    label,
                    f"both linked to {self.label(shared)}",
                    (seed_side[shared], candidate_side[shared]),
                    via=shared,
                    via_label=self.label(shared),
                )
            )
        return result


# ---------------------------------------------------------------- explainers


class Explainer:
    """Interface. `name` appears verbatim in run artifacts and paper tables."""

    name = "base"

    def explain(self, seed: str, candidate: str) -> Explanation:
        raise NotImplementedError


class MetaPathExplainer(Explainer):
    """Deterministic. No model, no sampling, no prompt — the control condition."""

    name = "metapath"

    def __init__(self, finder: MetaPathFinder, max_paths: int = 12):
        self.finder = finder
        self.max_paths = max_paths

    def explain(self, seed: str, candidate: str) -> Explanation:
        paths = self.finder.find(seed, candidate)[: self.max_paths]
        edge_ids: list[str] = []
        for path in paths:
            for edge_id in path.edge_ids:
                if edge_id not in edge_ids:
                    edge_ids.append(edge_id)

        if paths:
            text = "Connected by " + "; ".join(
                f"{p.relation.replace('_', ' ')} ({p.detail})" for p in paths
            ) + "."
        else:
            text = "No typed path was found between these papers in the graph."

        return Explanation(
            explainer=self.name,
            seed=seed,
            candidate=candidate,
            text=text,
            cited_edges=edge_ids,
            paths=list(paths),
            claims=[{"relation": p.relation, "detail": p.detail} for p in paths],
            diagnostics={"path_count": len(paths), "deterministic": True},
        )


_GROUNDED_SYSTEM = (
    "You explain why one scientific paper was retrieved for another. "
    "You will be given a numbered list of VERIFIED FACTS about how the two papers "
    "are connected in a citation graph. Use ONLY those facts. Do not use any "
    "outside knowledge about the papers, their authors, or their field. Do not "
    "speculate about content you were not given. "
    "Respond with a single JSON object and nothing else."
)

_GROUNDED_TEMPLATE = """Seed paper: {seed_title}
Candidate paper: {candidate_title}

VERIFIED FACTS (the only information you may use):
{facts}

Write a two-sentence explanation of why the candidate is relevant to the seed,
using only the facts above. Then list the fact numbers you actually relied on.

Return exactly this JSON shape:
{{"explanation": "<two sentences>", "fact_numbers": [<integers>]}}"""


class LLMGroundedExplainer(Explainer):
    """Prose from a model, but the model sees only extracted meta-path facts.

    It returns the FACT NUMBERS it relied on, and those map back to exact edge
    ids. So its citation is as precise as the metapath explainer's -- the only
    difference is that a model chose which subset to stand behind.
    """

    name = "llm-grounded"

    def __init__(self, finder: MetaPathFinder, client: OpenRouterClient, max_facts: int = 12):
        self.finder = finder
        self.client = client
        self.max_facts = max_facts

    def explain(self, seed: str, candidate: str) -> Explanation:
        paths = self.finder.find(seed, candidate)[: self.max_facts]
        seed_title = self.finder.label(seed)
        candidate_title = self.finder.label(candidate)

        if not paths:
            return Explanation(
                explainer=self.name,
                seed=seed,
                candidate=candidate,
                text="No typed path was found between these papers in the graph.",
                cited_edges=[],
                paths=[],
                diagnostics={"path_count": 0, "llm_called": False},
            )

        facts = "\n".join(
            f"{i + 1}. [{p.relation}] {p.detail}" for i, p in enumerate(paths)
        )
        prompt = _GROUNDED_TEMPLATE.format(
            seed_title=seed_title, candidate_title=candidate_title, facts=facts
        )

        try:
            parsed, response = self.client.complete_json(prompt, system=_GROUNDED_SYSTEM)
        except LLMUnavailable as exc:
            raise LLMUnavailable(f"{self.name} requires an LLM: {exc}") from exc

        diagnostics: dict[str, Any] = {
            "path_count": len(paths),
            "llm_called": True,
            "parse_ok": parsed is not None,
            **response.to_dict(),
        }

        if not parsed:
            # A model that cannot follow the schema gets NO edges, not all of
            # them. Defaulting to every fact would hand it the metapath
            # explainer's result and erase the difference we are measuring.
            diagnostics["failure"] = "unparseable_json"
            return Explanation(
                explainer=self.name,
                seed=seed,
                candidate=candidate,
                text=response.text.strip(),
                cited_edges=[],
                paths=list(paths),
                diagnostics=diagnostics,
            )

        # Accept anything int()-able (models emit "3" as often as 3), but record
        # every value that was offered and rejected. The two loops must use the
        # SAME coercion rule: a stricter rule here than in the diagnostic below
        # would hide schema violations from the count that exists to measure them.
        selected: list[int] = []
        out_of_range: list[int] = []
        unparseable = 0
        for value in parsed.get("fact_numbers") or []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                unparseable += 1
                continue
            if 1 <= number <= len(paths):
                selected.append(number)
            else:
                out_of_range.append(number)

        edge_ids: list[str] = []
        used_paths: list[MetaPath] = []
        for number in sorted(set(selected)):
            path = paths[number - 1]
            used_paths.append(path)
            for edge_id in path.edge_ids:
                if edge_id not in edge_ids:
                    edge_ids.append(edge_id)

        diagnostics["facts_offered"] = len(paths)
        diagnostics["facts_used"] = len(used_paths)
        diagnostics["hallucinated_fact_numbers"] = sorted(set(out_of_range))
        diagnostics["unparseable_fact_numbers"] = unparseable

        return Explanation(
            explainer=self.name,
            seed=seed,
            candidate=candidate,
            text=str(parsed.get("explanation") or response.text).strip(),
            cited_edges=edge_ids,
            paths=used_paths,
            claims=[{"relation": p.relation, "detail": p.detail} for p in used_paths],
            diagnostics=diagnostics,
        )


_FREE_SYSTEM = (
    "You explain why one scientific paper is relevant to another, based only on "
    "their titles and abstracts. Be specific about the relationship you believe "
    "exists between them. "
    "Respond with a single JSON object and nothing else."
)

_FREE_TEMPLATE = """Seed paper
Title: {seed_title}
Abstract: {seed_abstract}

Candidate paper
Title: {candidate_title}
Abstract: {candidate_abstract}

Explain in two sentences why the candidate paper is relevant to the seed paper.
Then state the specific relationships you believe connect them, choosing each
one from this list: direct_citation, co_citation, shared_author, shared_concept,
shared_venue.

Return exactly this JSON shape:
{{"explanation": "<two sentences>",
  "relationships": [{{"relation": "<one of the five>", "detail": "<what specifically>"}}]}}"""


class LLMFreeExplainer(Explainer):
    """Titles and abstracts only. The model never sees the graph.

    Its claimed relationships are grounded against the graph afterwards, in
    `_ground_claims`. Claims with no corresponding edge are recorded as
    `unrealised_claims` and cite nothing -- an explanation that names a
    relationship the graph does not contain cannot have caused the ranking.
    """

    name = "llm-free"

    def __init__(self, finder: MetaPathFinder, client: OpenRouterClient, db: Database):
        self.finder = finder
        self.client = client
        self.db = db

    def explain(self, seed: str, candidate: str) -> Explanation:
        seed_paper = self.db.get_paper(seed)
        candidate_paper = self.db.get_paper(candidate)
        if not seed_paper or not candidate_paper:
            return Explanation(
                explainer=self.name,
                seed=seed,
                candidate=candidate,
                text="Paper text unavailable.",
                diagnostics={"llm_called": False, "failure": "missing_paper_text"},
            )

        prompt = _FREE_TEMPLATE.format(
            seed_title=seed_paper["title"] or "(untitled)",
            seed_abstract=(seed_paper["abstract"] or "(no abstract)")[:2500],
            candidate_title=candidate_paper["title"] or "(untitled)",
            candidate_abstract=(candidate_paper["abstract"] or "(no abstract)")[:2500],
        )

        try:
            parsed, response = self.client.complete_json(prompt, system=_FREE_SYSTEM)
        except LLMUnavailable as exc:
            raise LLMUnavailable(f"{self.name} requires an LLM: {exc}") from exc

        diagnostics: dict[str, Any] = {
            "llm_called": True,
            "parse_ok": parsed is not None,
            "graph_shown_to_model": False,   # the guardrail, asserted in tests
            **response.to_dict(),
        }

        if not parsed:
            diagnostics["failure"] = "unparseable_json"
            return Explanation(
                explainer=self.name,
                seed=seed,
                candidate=candidate,
                text=response.text.strip(),
                cited_edges=[],
                diagnostics=diagnostics,
            )

        claims = []
        for entry in parsed.get("relationships") or []:
            if not isinstance(entry, Mapping):
                continue
            relation = str(entry.get("relation") or "").strip().lower()
            claims.append({"relation": relation, "detail": str(entry.get("detail") or "")})

        realised, unrealised, used_paths = self._ground_claims(seed, candidate, claims)

        diagnostics["claims_made"] = len(claims)
        diagnostics["claims_realised"] = len(claims) - len(unrealised)
        diagnostics["claims_unrealised"] = len(unrealised)

        return Explanation(
            explainer=self.name,
            seed=seed,
            candidate=candidate,
            text=str(parsed.get("explanation") or response.text).strip(),
            cited_edges=realised,
            paths=used_paths,
            claims=claims,
            unrealised_claims=unrealised,
            diagnostics=diagnostics,
        )

    def _ground_claims(
        self, seed: str, candidate: str, claims: Sequence[Mapping[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]], list[MetaPath]]:
        """Map claimed relation types onto real edges. Runs AFTER the model call.

        Grounding is by relation TYPE, not by matching the model's free-text
        detail against node labels. Matching on detail text would let a vague
        claim collect every edge of that type by accident, which would inflate
        this explainer's apparent faithfulness.
        """
        available = self.finder.find(seed, candidate)
        by_relation: dict[str, list[MetaPath]] = {}
        for path in available:
            by_relation.setdefault(path.relation, []).append(path)

        edge_ids: list[str] = []
        used_paths: list[MetaPath] = []
        unrealised: list[dict[str, Any]] = []

        for claim in claims:
            relation = claim.get("relation")
            if relation not in RELATION_TYPES or relation not in by_relation:
                unrealised.append(
                    {
                        **claim,
                        "reason": "no such connection exists in the graph"
                        if relation in RELATION_TYPES
                        else "not a recognised relation type",
                    }
                )
                continue
            for path in by_relation[relation]:
                if path not in used_paths:
                    used_paths.append(path)
                for edge_id in path.edge_ids:
                    if edge_id not in edge_ids:
                        edge_ids.append(edge_id)

        return edge_ids, unrealised, used_paths


# ----------------------------------------------------------------- registry


def build_explainers(
    graph: TypedGraph,
    db: Database,
    client: OpenRouterClient | None,
    max_paths: int = 12,
    names: Iterable[str] | None = None,
) -> dict[str, Explainer]:
    """Instantiate the requested explainers.

    LLM-backed explainers are omitted with a warning rather than stubbed when no
    key is present, so a run can never report a metapath result under an LLM
    label.
    """
    finder = MetaPathFinder(graph, db, max_paths=max_paths)
    wanted = set(names) if names else {"metapath", "llm-grounded", "llm-free"}
    explainers: dict[str, Explainer] = {}

    if "metapath" in wanted:
        explainers["metapath"] = MetaPathExplainer(finder, max_paths=max_paths)

    needs_llm = wanted & {"llm-grounded", "llm-free"}
    if needs_llm:
        if client is None or not client.available:
            log.warning(
                "skipping %s: OPENROUTER_API_KEY is not set",
                ", ".join(sorted(needs_llm)),
            )
        else:
            if "llm-grounded" in wanted:
                explainers["llm-grounded"] = LLMGroundedExplainer(finder, client, max_paths)
            if "llm-free" in wanted:
                explainers["llm-free"] = LLMFreeExplainer(finder, client, db)

    return explainers
