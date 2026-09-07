"""Phase 6a — derive the prerequisite DAG.

The reading tree's layout is a research claim, so it is DERIVED here in Python
and never authored by an LLM. An LLM-written tree is unfalsifiable: you cannot
ablate a prerequisite edge that exists only because a model asserted it.

Edge rule -- A is a prerequisite of B if any of:
  1. B cites A directly AND A predates B.
  2. A and B share a concept AND A has materially higher citation count
     (A is the canonical statement of that concept).
  3. A appears in B's related-work section (needs S2ORC full text).

Rule 3 is implemented but inert on this corpus: neither OpenAlex nor the S2
Graph API serves section-segmented full text, so `related_work_available` is
False in every artifact and the rule contributes no edges. That is recorded
rather than quietly skipped -- it belongs in the paper's limitations.

Cycles are real in citation graphs (preprint/published pairs, simultaneous
submissions). Every cycle-breaking drop is logged into the artifact, because
silently discarding them would hide a genuine limitation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .db import Database
from .graph import TypedGraph

log = logging.getLogger(__name__)


REASON_CITATION = "cites_and_predates"
REASON_CANONICAL = "canonical_for_shared_concept"
REASON_RELATED_WORK = "appears_in_related_work"


@dataclass
class PrereqEdge:
    prereq: str
    dependent: str
    reason: str
    weight: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prereq": self.prereq,
            "dependent": self.dependent,
            "reason": self.reason,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass
class CycleDrop:
    """One edge removed to make the graph acyclic. Always reported."""

    prereq: str
    dependent: str
    reason: str
    weight: float
    cycle: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_edge": {"prereq": self.prereq, "dependent": self.dependent},
            "reason": self.reason,
            "weight": self.weight,
            "cycle": self.cycle,
        }


# ------------------------------------------------------------- edge derivation


def derive_prereq_edges(
    db: Database,
    graph: TypedGraph,
    paper_ids: Sequence[str],
    citation_dominance_ratio: float = 3.0,
    related_work: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[PrereqEdge], dict[str, Any]]:
    """Apply the three prerequisite rules over a candidate paper set."""
    members = list(dict.fromkeys(paper_ids))
    member_set = set(members)
    papers = {pid: db.get_paper(pid) for pid in members}
    papers = {pid: p for pid, p in papers.items() if p}

    edges: dict[tuple[str, str], PrereqEdge] = {}
    counts = {REASON_CITATION: 0, REASON_CANONICAL: 0, REASON_RELATED_WORK: 0}
    undated_pairs = 0

    # --- rule 1: direct citation with a date ordering -----------------------
    for dependent in members:
        for rel, neighbour, _edge_id in graph.neighbours(dependent, ["cites"]):
            if neighbour not in member_set:
                continue
            a, b = papers.get(neighbour), papers.get(dependent)
            if not a or not b:
                continue
            if a["year"] is None or b["year"] is None:
                # Without dates the citation direction alone cannot establish
                # precedence, so the pair is skipped and counted rather than
                # assumed. Assuming would invent tiers out of missing metadata.
                undated_pairs += 1
                continue
            if a["year"] < b["year"]:
                key = (neighbour, dependent)
                if key not in edges:
                    edges[key] = PrereqEdge(
                        prereq=neighbour,
                        dependent=dependent,
                        reason=REASON_CITATION,
                        weight=3.0,
                        detail=f"{b['year']} paper cites {a['year']} paper",
                    )
                    counts[REASON_CITATION] += 1

    # --- rule 2: canonical statement of a shared concept --------------------
    concept_members: dict[str, list[str]] = defaultdict(list)
    for pid in members:
        for _rel, concept, _edge_id in graph.neighbours(pid, ["shares_concept"]):
            concept_members[concept].append(pid)

    for concept, holders in concept_members.items():
        if len(holders) < 2:
            continue
        for a_id in holders:
            for b_id in holders:
                if a_id == b_id:
                    continue
                a, b = papers.get(a_id), papers.get(b_id)
                if not a or not b:
                    continue
                a_cites = a["citation_count"] or 0
                b_cites = b["citation_count"] or 0
                if b_cites <= 0:
                    continue
                if a_cites < citation_dominance_ratio * b_cites:
                    continue
                # A canonical paper published AFTER its dependent is not a
                # prerequisite, however dominant its citation count.
                if a["year"] is not None and b["year"] is not None and a["year"] > b["year"]:
                    continue
                key = (a_id, b_id)
                if key not in edges:
                    label = db.get_node(concept)
                    edges[key] = PrereqEdge(
                        prereq=a_id,
                        dependent=b_id,
                        reason=REASON_CANONICAL,
                        weight=1.0,
                        detail=(
                            f"canonical for '{(label or {}).get('label', concept)}' "
                            f"({a_cites} vs {b_cites} citations)"
                        ),
                    )
                    counts[REASON_CANONICAL] += 1

    # --- rule 3: related-work membership (needs full text) ------------------
    if related_work:
        for dependent, prereqs in related_work.items():
            if dependent not in member_set:
                continue
            for prereq in prereqs:
                if prereq not in member_set or prereq == dependent:
                    continue
                key = (prereq, dependent)
                if key not in edges:
                    edges[key] = PrereqEdge(
                        prereq=prereq,
                        dependent=dependent,
                        reason=REASON_RELATED_WORK,
                        weight=2.0,
                        detail="cited in the related-work section",
                    )
                    counts[REASON_RELATED_WORK] += 1

    diagnostics = {
        "rule_counts": counts,
        "undated_citation_pairs_skipped": undated_pairs,
        "related_work_available": bool(related_work),
        "citation_dominance_ratio": citation_dominance_ratio,
        "candidate_papers": len(members),
    }
    return list(edges.values()), diagnostics


# ------------------------------------------------------------ cycle breaking


def break_cycles(edges: Sequence[PrereqEdge]) -> tuple[list[PrereqEdge], list[CycleDrop]]:
    """Remove the lowest-weight edge of each cycle until the graph is acyclic.

    Every drop is returned. Citation graphs genuinely contain cycles -- a
    preprint and its published version cite each other, simultaneous submissions
    cross-reference -- and pretending otherwise would hide a real limitation.
    """
    import networkx as nx

    digraph = nx.DiGraph()
    lookup: dict[tuple[str, str], PrereqEdge] = {}
    for edge in edges:
        key = (edge.prereq, edge.dependent)
        lookup[key] = edge
        digraph.add_edge(edge.prereq, edge.dependent, weight=edge.weight)

    drops: list[CycleDrop] = []
    # Bounded to avoid pathological looping on a dense component; each pass
    # removes at least one edge, so this terminates well before the bound.
    for _ in range(len(edges) + 1):
        try:
            cycle = nx.find_cycle(digraph, orientation="original")
        except nx.NetworkXNoCycle:
            break

        cycle_edges = [(u, v) for u, v, *_ in cycle]
        weakest = min(cycle_edges, key=lambda e: (lookup[e].weight, e[0], e[1]))
        edge = lookup[weakest]
        drops.append(
            CycleDrop(
                prereq=edge.prereq,
                dependent=edge.dependent,
                reason=edge.reason,
                weight=edge.weight,
                cycle=[u for u, _v in cycle_edges],
            )
        )
        digraph.remove_edge(*weakest)
        log.info("cycle broken: dropped %s -> %s (weight %.1f)",
                 edge.prereq, edge.dependent, edge.weight)

    kept = [
        lookup[(u, v)] for u, v in digraph.edges() if (u, v) in lookup
    ]
    return kept, drops


def transitive_reduction(
    edges: Sequence[PrereqEdge],
) -> tuple[list[PrereqEdge], list[PrereqEdge]]:
    """Drop prerequisite edges already implied by a longer path.

    If A is a prerequisite of B and B of C, then the explicit A->C edge tells the
    reader nothing they cannot follow through B, but it costs a wire across the
    canvas. Removing it is information-preserving: reachability is identical, so
    every "you must read A before C" relation survives.

    Added after the Phase 6c design critique measured 58 wires over 15 nodes
    (3.9 per node) and 343 over 40 (8.6 per node). The canvas was legible only
    because the tier cap kept node counts low, which is the wrong reason for a
    layout to work.

    Returns (kept, removed). Removed edges are reported in the artifact so the
    reduction is auditable rather than invisible.
    """
    import networkx as nx

    digraph = nx.DiGraph()
    lookup: dict[tuple[str, str], PrereqEdge] = {}
    for edge in edges:
        key = (edge.prereq, edge.dependent)
        lookup[key] = edge
        digraph.add_edge(*key)

    if not digraph or not nx.is_directed_acyclic_graph(digraph):
        # Only valid on a DAG; callers run this after break_cycles, but a
        # surprise cycle must not silently corrupt the layout.
        log.warning("skipping transitive reduction: graph is not acyclic")
        return list(edges), []

    reduced = nx.transitive_reduction(digraph)
    kept_keys = set(reduced.edges())

    kept = [lookup[k] for k in sorted(kept_keys) if k in lookup]
    removed = [lookup[k] for k in sorted(lookup) if k not in kept_keys]
    if removed:
        log.info("transitive reduction removed %d implied prerequisite edges", len(removed))
    return kept, removed


# ----------------------------------------------------------- tier assignment


def assign_tiers(
    node_ids: Sequence[str], edges: Sequence[PrereqEdge], max_tiers: int = 6
) -> dict[str, int]:
    """Longest-path depth from any root. Assumes an acyclic edge set."""
    incoming: dict[str, list[str]] = {n: [] for n in node_ids}
    outgoing: dict[str, list[str]] = {n: [] for n in node_ids}
    for edge in edges:
        if edge.prereq in incoming and edge.dependent in incoming:
            incoming[edge.dependent].append(edge.prereq)
            outgoing[edge.prereq].append(edge.dependent)

    depth = {n: 0 for n in node_ids}
    # Kahn ordering, then relax forward: longest path is well defined on a DAG.
    in_degree = {n: len(incoming[n]) for n in node_ids}
    queue = [n for n in node_ids if in_degree[n] == 0]
    order: list[str] = []
    while queue:
        queue.sort()  # deterministic
        node = queue.pop(0)
        order.append(node)
        for child in outgoing[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    for node in order:
        for child in outgoing[node]:
            depth[child] = max(depth[child], depth[node] + 1)

    if max_tiers > 0:
        # Compress rather than truncate: a node beyond the tier cap still needs a
        # column, and clamping keeps its prerequisites to its left.
        for node in depth:
            depth[node] = min(depth[node], max_tiers - 1)
    return depth


def tier_labels(
    tiers: Mapping[str, int], papers: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Label each tier from the actual years of the papers in it.

    Derived, not authored: the label reports what is in the column rather than
    asserting a narrative about it.
    """
    by_tier: dict[int, list[int]] = defaultdict(list)
    for node_id, tier in tiers.items():
        year = (papers.get(node_id) or {}).get("year")
        if year:
            by_tier[tier].append(int(year))

    if not tiers:
        return []

    labels = []
    for tier in range(max(tiers.values()) + 1):
        years = sorted(by_tier.get(tier, []))
        if not years:
            labels.append(f"Tier {tier + 1}")
        elif years[0] == years[-1]:
            labels.append(f"Tier {tier + 1} · {years[0]}")
        else:
            labels.append(f"Tier {tier + 1} · {years[0]}–{years[-1]}")
    return labels


# -------------------------------------------------------------------- build


def build_tree(
    db: Database,
    graph: TypedGraph,
    seed: str,
    candidates: Sequence[Mapping[str, Any]],
    necessity: Mapping[str, float] | None = None,
    verdicts: Mapping[str, Mapping[str, Any]] | None = None,
    why: Mapping[str, Sequence[tuple[str, str]]] | None = None,
    max_nodes: int = 15,
    max_tiers: int = 6,
    citation_dominance_ratio: float = 3.0,
    related_work: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Produce the tree artifact for one seed.

    `candidates` are the ranked retrieval results. Pruning to `max_nodes` is by
    Necessity Score, so the surviving nodes are the ones whose placement in the
    tree is causally justified -- not merely the highest ranked.
    """
    necessity = necessity or {}
    verdicts = verdicts or {}
    why = why or {}

    candidate_ids = [c["paper_id"] for c in candidates if c["paper_id"] != seed]

    # --- prune FIRST, then derive structure over the survivors --------------
    # Deriving over all candidates and pruning afterwards would leave dangling
    # prerequisites pointing at removed nodes.
    keep_budget = max(1, max_nodes - 1)  # the seed always occupies one slot
    ranked = sorted(
        candidate_ids,
        key=lambda pid: (-float(necessity.get(pid, 0.0)), _rank_of(candidates, pid)),
    )
    kept = ranked[:keep_budget]
    pruned = ranked[keep_budget:]

    members = [seed] + kept
    papers = {pid: db.get_paper(pid) for pid in members}
    papers = {pid: p for pid, p in papers.items() if p}
    members = [m for m in members if m in papers]

    edges, edge_diagnostics = derive_prereq_edges(
        db, graph, members, citation_dominance_ratio, related_work
    )
    acyclic_edges, drops = break_cycles(edges)
    # Tiers are computed from the FULL acyclic edge set, before reduction.
    # Reduction preserves reachability but not path length, so reducing first
    # would shorten longest paths and collapse the tier structure.
    tiers = assign_tiers(members, acyclic_edges, max_tiers)
    display_edges, implied = transitive_reduction(acyclic_edges)

    reqs: dict[str, list[list[str]]] = defaultdict(list)
    for edge in display_edges:
        reqs[edge.dependent].append([edge.prereq, edge.detail or edge.reason])

    # Row index within a tier: citation count desc, then id, so the layout is
    # stable across runs and a reader can find a paper where they left it.
    rows: dict[str, int] = {}
    by_tier: dict[int, list[str]] = defaultdict(list)
    for node_id in members:
        by_tier[tiers.get(node_id, 0)].append(node_id)
    for tier, node_ids in by_tier.items():
        node_ids.sort(key=lambda pid: (-(papers[pid]["citation_count"] or 0), pid))
        for row, node_id in enumerate(node_ids):
            rows[node_id] = row

    nodes = []
    for node_id in members:
        paper = papers[node_id]
        nodes.append(
            {
                "id": node_id,
                "tier": tiers.get(node_id, 0),
                "row": rows.get(node_id, 0),
                "kind": "seed" if node_id == seed else "candidate",
                "title": paper["title"],
                "year": paper["year"],
                "cites": paper["citation_count"],
                "abstract": paper["abstract"],
                "reqs": sorted(reqs.get(node_id, []), key=lambda r: r[0]),
                "why": [list(w) for w in why.get(node_id, [])],
                "necessity": float(necessity.get(node_id, 0.0)),
                "verdict": dict(verdicts[node_id]) if node_id in verdicts else None,
            }
        )
    nodes.sort(key=lambda n: (n["tier"], n["row"]))

    return {
        "seed": seed,
        "nodes": nodes,
        "neighbourhood": export_neighbourhood(db, graph, members),
        "tiers": tier_labels(tiers, papers),
        "prereq_edges": [e.to_dict() for e in display_edges],
        "implied_edges_removed": [e.to_dict() for e in implied],
        "cycle_drops": [d.to_dict() for d in drops],
        "diagnostics": {
            **edge_diagnostics,
            "prereq_edges_derived": len(acyclic_edges),
            "prereq_edges_displayed": len(display_edges),
            "implied_edges_removed": len(implied),
            "edges_per_node": (
                round(len(display_edges) / len(members), 2) if members else 0.0
            ),
            "canvas_legible": _canvas_verdict(members, tiers, display_edges),
            "nodes_kept": len(members),
            "nodes_pruned": len(pruned),
            "pruned_ids": pruned,
            "pruned_by": "necessity_score",
            "cycles_broken": len(drops),
            "max_nodes": max_nodes,
            "max_tiers": max_tiers,
        },
    }


# Thresholds measured during the Phase 6c design critique on the 500-paper
# corpus. At 40 nodes the canvas put 19 rows in a single tier and drew 8.6 wires
# per node -- a hairball in which the tier abstraction has stopped meaning
# anything. These are the numbers that justify `tree.max_nodes: 15`.
MAX_LEGIBLE_ROWS_PER_TIER = 6
MAX_LEGIBLE_EDGES_PER_NODE = 4.0


def _canvas_verdict(
    members: Sequence[str], tiers: Mapping[str, int], edges: Sequence[PrereqEdge]
) -> dict[str, Any]:
    """Report whether this tree is actually readable at the configured cap.

    Surfaced in the artifact rather than merely logged, so raising `max_nodes`
    produces visible evidence of the cost instead of a quietly worse canvas.
    """
    rows_per_tier: dict[int, int] = defaultdict(int)
    for node_id in members:
        rows_per_tier[tiers.get(node_id, 0)] += 1
    max_rows = max(rows_per_tier.values()) if rows_per_tier else 0
    per_node = len(edges) / len(members) if members else 0.0

    reasons = []
    if max_rows > MAX_LEGIBLE_ROWS_PER_TIER:
        reasons.append(f"{max_rows} rows stacked in one tier")
    if per_node > MAX_LEGIBLE_EDGES_PER_NODE:
        reasons.append(f"{per_node:.1f} prerequisite wires per node")

    return {
        "ok": not reasons,
        "max_rows_in_a_tier": max_rows,
        "edges_per_node": round(per_node, 2),
        "reasons": reasons,
    }


def export_neighbourhood(
    db: Database,
    graph: TypedGraph,
    paper_ids: Sequence[str],
    max_satellites_per_kind: int = 60,
) -> dict[str, Any]:
    """Typed subgraph around the tree's papers, for the neuron-graph view.

    Papers become soma; authors, venues and concepts become the smaller boutons.
    Satellite nodes touching only ONE paper are dropped: they cannot lie on a
    path between two papers, so they add visual mass without adding structure.
    The cap is per kind so a paper with 200 authors cannot crowd out every
    concept node.
    """
    members = set(paper_ids)
    satellite_degree: dict[str, int] = defaultdict(int)
    satellite_kind: dict[str, str] = {}
    raw_edges: list[tuple[str, str, str, str]] = []

    for pid in paper_ids:
        for rel, neighbour, edge_id in graph.neighbours(pid):
            if rel in ("cites", "cited_by"):
                if neighbour in members and rel == "cites":
                    raw_edges.append((edge_id, pid, "cites", neighbour))
                continue
            if neighbour in members:
                continue
            kind = graph.node_kind(neighbour)
            if kind == "paper":
                continue
            satellite_degree[neighbour] += 1
            satellite_kind[neighbour] = kind
            raw_edges.append((edge_id, pid, rel, neighbour))

    keep: set[str] = set()
    by_kind: dict[str, list[str]] = defaultdict(list)
    for node_id, degree in satellite_degree.items():
        if degree >= 2:
            by_kind[satellite_kind[node_id]].append(node_id)
    for kind, node_ids in by_kind.items():
        node_ids.sort(key=lambda n: (-satellite_degree[n], n))
        keep.update(node_ids[:max_satellites_per_kind])

    nodes = []
    for pid in paper_ids:
        paper = db.get_paper(pid)
        nodes.append(
            {
                "id": pid,
                "kind": "paper",
                "label": (paper or {}).get("title") or pid,
                "cites": (paper or {}).get("citation_count") or 0,
            }
        )
    for node_id in sorted(keep):
        node = db.get_node(node_id) or {}
        nodes.append(
            {
                "id": node_id,
                "kind": satellite_kind[node_id],
                "label": node.get("label") or node_id,
                "degree": satellite_degree[node_id],
            }
        )

    present = {n["id"] for n in nodes}
    edges = sorted(
        {
            (edge_id, src, rel, dst)
            for edge_id, src, rel, dst in raw_edges
            if src in present and dst in present
        }
    )
    return {
        "nodes": nodes,
        "edges": [
            {"edge_id": e, "src": s, "rel": r, "dst": d} for e, s, r, d in edges
        ],
    }


def _rank_of(candidates: Sequence[Mapping[str, Any]], paper_id: str) -> int:
    for candidate in candidates:
        if candidate["paper_id"] == paper_id:
            return int(candidate.get("rank") or 0)
    return 10**6


def write_tree(tree: Mapping[str, Any], runs_dir: str | Path) -> Path:
    path = Path(runs_dir) / f"tree_{tree['seed']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    return path
