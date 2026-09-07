"""Tests for the prerequisite DAG (Phase 6a).

The tree's layout is a research claim, so its derivation needs the same scrutiny
as the ablation harness. The rules under test:
  1. B cites A and A predates B
  2. A and B share a concept and A dominates in citations
  3. A appears in B's related work (inert without full text)
plus cycle breaking, transitive reduction, tiering, and necessity-based pruning.
"""

from __future__ import annotations

import pytest

from synapse.tree import (
    MAX_LEGIBLE_EDGES_PER_NODE,
    PrereqEdge,
    REASON_CANONICAL,
    REASON_CITATION,
    REASON_RELATED_WORK,
    assign_tiers,
    break_cycles,
    build_tree,
    derive_prereq_edges,
    export_neighbourhood,
    tier_labels,
    transitive_reduction,
)


def edge(prereq, dependent, reason=REASON_CITATION, weight=3.0):
    return PrereqEdge(prereq=prereq, dependent=dependent, reason=reason, weight=weight)


# ------------------------------------------------------------ rule 1: cites


def test_citation_edge_requires_the_prerequisite_to_predate(corpus, graph):
    """P_SEED (2021) cites P_FAITH (2019), so P_FAITH is the prerequisite."""
    edges, _ = derive_prereq_edges(corpus, graph, ["P_SEED", "P_FAITH"])

    citation_edges = [e for e in edges if e.reason == REASON_CITATION]
    assert any(e.prereq == "P_FAITH" and e.dependent == "P_SEED" for e in citation_edges)


def test_citation_edge_is_not_created_in_the_wrong_time_direction(corpus, graph):
    edges, _ = derive_prereq_edges(corpus, graph, ["P_SEED", "P_FAITH"])

    assert not any(
        e.prereq == "P_SEED" and e.dependent == "P_FAITH" and e.reason == REASON_CITATION
        for e in edges
    )


def test_undated_pairs_are_skipped_and_counted(corpus, graph):
    """Missing years must not be used to invent an ordering."""
    corpus.upsert_paper({
        "paper_id": "P_NOYEAR", "title": "No year", "abstract": "x" * 100,
        "year": None, "venue": "SIGIR", "citation_count": 5, "reference_count": 0,
        "fields": [], "source": "synthetic",
    })
    corpus.add_edge("P_SEED", "cites", "P_NOYEAR")
    from synapse.graph import TypedGraph
    fresh = TypedGraph.from_db(corpus)

    edges, diagnostics = derive_prereq_edges(corpus, fresh, ["P_SEED", "P_NOYEAR"])

    assert diagnostics["undated_citation_pairs_skipped"] >= 1
    assert not any(e.dependent == "P_SEED" and e.prereq == "P_NOYEAR" for e in edges)


# -------------------------------------------------------- rule 2: canonical


def test_canonical_concept_edge_requires_citation_dominance(corpus, graph):
    """P_REF2 (500 cites) dominates P_UNFAITH (5) — but they must share a concept."""
    edges, _ = derive_prereq_edges(
        corpus, graph, ["P_SEED", "P_UNFAITH", "P_REF1", "P_REF2"],
        citation_dominance_ratio=3.0,
    )

    canonical = [e for e in edges if e.reason == REASON_CANONICAL]
    for e in canonical:
        prereq = corpus.get_paper(e.prereq)
        dependent = corpus.get_paper(e.dependent)
        assert prereq["citation_count"] >= 3.0 * dependent["citation_count"]


def test_raising_the_dominance_ratio_removes_canonical_edges(corpus, graph):
    lenient, _ = derive_prereq_edges(corpus, graph, ["P_SEED", "P_UNFAITH", "P_CITER"],
                                     citation_dominance_ratio=1.1)
    strict, _ = derive_prereq_edges(corpus, graph, ["P_SEED", "P_UNFAITH", "P_CITER"],
                                    citation_dominance_ratio=1000.0)

    lenient_count = sum(1 for e in lenient if e.reason == REASON_CANONICAL)
    strict_count = sum(1 for e in strict if e.reason == REASON_CANONICAL)
    assert strict_count < lenient_count


def test_a_canonical_paper_published_later_is_not_a_prerequisite(corpus, graph):
    edges, _ = derive_prereq_edges(corpus, graph, ["P_SEED", "P_UNFAITH", "P_CITER"])

    for e in edges:
        prereq = corpus.get_paper(e.prereq)
        dependent = corpus.get_paper(e.dependent)
        if prereq["year"] and dependent["year"]:
            assert prereq["year"] <= dependent["year"]


# ---------------------------------------------------- rule 3: related work


def test_related_work_rule_is_inert_without_full_text(corpus, graph):
    _edges, diagnostics = derive_prereq_edges(corpus, graph, ["P_SEED", "P_FAITH"])

    assert diagnostics["related_work_available"] is False
    assert diagnostics["rule_counts"][REASON_RELATED_WORK] == 0


def test_related_work_rule_creates_edges_when_full_text_is_supplied(corpus, graph):
    edges, diagnostics = derive_prereq_edges(
        corpus, graph, ["P_SEED", "P_UNFAITH"],
        related_work={"P_SEED": ["P_UNFAITH"]},
    )

    assert diagnostics["related_work_available"] is True
    assert any(e.reason == REASON_RELATED_WORK for e in edges)


# ------------------------------------------------------------ cycle breaking


def test_cycles_are_broken_and_every_drop_is_reported():
    edges = [edge("A", "B", weight=3.0), edge("B", "C", weight=3.0),
             edge("C", "A", weight=1.0)]

    kept, drops = break_cycles(edges)

    assert len(drops) == 1
    assert (drops[0].prereq, drops[0].dependent) == ("C", "A")
    assert len(kept) == 2


def test_cycle_breaking_drops_the_lowest_weight_edge():
    edges = [edge("A", "B", weight=1.0), edge("B", "A", weight=3.0)]

    _kept, drops = break_cycles(edges)

    assert (drops[0].prereq, drops[0].dependent) == ("A", "B")


def test_acyclic_input_is_left_untouched():
    edges = [edge("A", "B"), edge("B", "C")]

    kept, drops = break_cycles(edges)

    assert drops == []
    assert len(kept) == 2


def test_cycle_drop_records_the_cycle_it_came_from():
    _kept, drops = break_cycles([edge("A", "B", weight=1.0), edge("B", "A", weight=2.0)])

    assert drops[0].cycle


# ------------------------------------------------------ transitive reduction


def test_transitive_reduction_removes_the_implied_shortcut():
    """A→B→C makes A→C redundant: reachability is unchanged."""
    edges = [edge("A", "B"), edge("B", "C"), edge("A", "C")]

    kept, removed = transitive_reduction(edges)

    assert len(kept) == 2
    assert [(e.prereq, e.dependent) for e in removed] == [("A", "C")]


def test_transitive_reduction_preserves_reachability():
    import networkx as nx

    edges = [edge("A", "B"), edge("B", "C"), edge("A", "C"), edge("A", "D"), edge("C", "D")]
    original = nx.DiGraph([(e.prereq, e.dependent) for e in edges])

    kept, _removed = transitive_reduction(edges)
    reduced = nx.DiGraph([(e.prereq, e.dependent) for e in kept])

    for node in original.nodes:
        assert nx.descendants(original, node) == nx.descendants(reduced, node)


def test_transitive_reduction_keeps_an_already_minimal_chain():
    kept, removed = transitive_reduction([edge("A", "B"), edge("B", "C")])

    assert len(kept) == 2 and removed == []


def test_transitive_reduction_refuses_a_cyclic_graph():
    edges = [edge("A", "B"), edge("B", "A")]

    kept, removed = transitive_reduction(edges)

    assert len(kept) == 2 and removed == []


# ------------------------------------------------------------------- tiers


def test_tier_is_the_longest_path_not_the_shortest():
    """D sits at tier 3 via A→B→C→D even though A→D exists."""
    nodes = ["A", "B", "C", "D"]
    edges = [edge("A", "B"), edge("B", "C"), edge("C", "D"), edge("A", "D")]

    tiers = assign_tiers(nodes, edges, max_tiers=10)

    assert tiers["A"] == 0 and tiers["D"] == 3


def test_roots_sit_in_tier_zero():
    tiers = assign_tiers(["A", "B"], [edge("A", "B")], max_tiers=6)

    assert tiers["A"] == 0


def test_tiers_are_clamped_to_the_configured_maximum():
    nodes = list("ABCDEFGH")
    edges = [edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    tiers = assign_tiers(nodes, edges, max_tiers=3)

    assert max(tiers.values()) == 2


def test_isolated_nodes_are_tier_zero():
    tiers = assign_tiers(["A", "B"], [], max_tiers=6)

    assert tiers == {"A": 0, "B": 0}


def test_tier_labels_are_derived_from_actual_years():
    papers = {"A": {"year": 2015}, "B": {"year": 2020}, "C": {"year": 2021}}

    labels = tier_labels({"A": 0, "B": 1, "C": 1}, papers)

    assert labels[0] == "Tier 1 · 2015"
    assert "2020" in labels[1] and "2021" in labels[1]


# ------------------------------------------------------------- build_tree


@pytest.fixture
def candidates(corpus):
    ids = [p["paper_id"] for p in corpus.all_papers() if p["paper_id"] != "P_SEED"]
    return [{"paper_id": pid, "rank": i + 1} for i, pid in enumerate(sorted(ids))]


def test_build_tree_caps_the_node_count(corpus, graph, candidates):
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=6)

    assert len(tree["nodes"]) <= 6


def test_build_tree_prunes_by_necessity_score(corpus, graph, candidates):
    """The surviving nodes must be the causally justified ones, not the top ranked."""
    necessity = {c["paper_id"]: 0.0 for c in candidates}
    necessity["P_UNFAITH"] = 0.9
    necessity["P_REF2"] = 0.8

    tree = build_tree(corpus, graph, "P_SEED", candidates, necessity=necessity, max_nodes=3)

    kept = {n["id"] for n in tree["nodes"]}
    assert "P_UNFAITH" in kept and "P_REF2" in kept


def test_build_tree_always_includes_the_seed(corpus, graph, candidates):
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=2)

    assert any(n["kind"] == "seed" and n["id"] == "P_SEED" for n in tree["nodes"])


def test_no_prerequisite_points_at_a_pruned_node(corpus, graph, candidates):
    """Deriving before pruning would leave prerequisites dangling."""
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=5)

    present = {n["id"] for n in tree["nodes"]}
    for node in tree["nodes"]:
        for prereq_id, _reason in node["reqs"]:
            assert prereq_id in present


def test_nodes_carry_the_verdict_they_were_given(corpus, graph, candidates):
    verdicts = {"P_FAITH": {"ok": True, "from": 1, "to": 9, "control": 2.0, "note": "n"}}

    tree = build_tree(corpus, graph, "P_SEED", candidates, verdicts=verdicts, max_nodes=15)

    node = next(n for n in tree["nodes"] if n["id"] == "P_FAITH")
    assert node["verdict"]["ok"] is True


def test_tree_reports_whether_the_canvas_is_legible(corpus, graph, candidates):
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=15)
    verdict = tree["diagnostics"]["canvas_legible"]

    assert set(verdict) == {"ok", "max_rows_in_a_tier", "edges_per_node", "reasons"}
    if verdict["ok"]:
        assert verdict["edges_per_node"] <= MAX_LEGIBLE_EDGES_PER_NODE


def test_tree_reports_the_edges_reduction_removed(corpus, graph, candidates):
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=15)

    diagnostics = tree["diagnostics"]
    assert diagnostics["prereq_edges_displayed"] <= diagnostics["prereq_edges_derived"]
    assert len(tree["implied_edges_removed"]) == diagnostics["implied_edges_removed"]


def test_rows_are_unique_within_a_tier(corpus, graph, candidates):
    tree = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=15)

    seen = set()
    for node in tree["nodes"]:
        key = (node["tier"], node["row"])
        assert key not in seen, f"two nodes share slot {key}"
        seen.add(key)


def test_build_tree_is_deterministic(corpus, graph, candidates):
    first = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=8)
    second = build_tree(corpus, graph, "P_SEED", candidates, max_nodes=8)

    assert [n["id"] for n in first["nodes"]] == [n["id"] for n in second["nodes"]]
    assert first["tiers"] == second["tiers"]


# ------------------------------------------------------------ neighbourhood


def test_neighbourhood_export_includes_typed_satellites(corpus, graph):
    hood = export_neighbourhood(corpus, graph, ["P_SEED", "P_FAITH", "P_UNFAITH"])

    kinds = {n["kind"] for n in hood["nodes"]}
    assert "paper" in kinds
    assert kinds & {"author", "concept", "venue"}


def test_neighbourhood_drops_satellites_touching_only_one_paper(corpus, graph):
    """A satellite on one paper cannot lie on a path between two."""
    hood = export_neighbourhood(corpus, graph, ["P_SEED", "P_FAITH"])

    counts = {}
    for e in hood["edges"]:
        if e["rel"] != "cites":
            counts[e["dst"]] = counts.get(e["dst"], 0) + 1
    assert all(count >= 2 for count in counts.values())


def test_neighbourhood_edges_only_reference_present_nodes(corpus, graph):
    hood = export_neighbourhood(corpus, graph, ["P_SEED", "P_FAITH", "P_REF1"])

    present = {n["id"] for n in hood["nodes"]}
    for e in hood["edges"]:
        assert e["src"] in present and e["dst"] in present
