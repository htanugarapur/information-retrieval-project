"""Tests for the claim the whole paper rests on: recorded edge flows are exact.

If these fail, `contributing_edges` is approximate and the ablation protocol
measures nothing. They are deliberately the strictest tests in the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from synapse.graph import TypedGraph
from synapse.ppr import (
    FlowIndex,
    attribute_edges,
    personalized_pagerank,
    reference_flow_accumulation,
)
from synapse.db import make_edge_id


def build_graph(edges, nodes=None):
    if nodes is None:
        collected = {}
        for src, rel, dst in edges:
            collected[src] = "paper"
            collected[dst] = "paper"
        nodes = sorted(collected.items())
    edge_rows = [
        {
            "edge_id": make_edge_id(src, rel, dst),
            "src": src,
            "rel": rel,
            "dst": dst,
            "weight": 1.0,
            "attrs": {},
        }
        for src, rel, dst in edges
    ]
    return TypedGraph.from_rows(nodes, edge_rows)


@pytest.fixture
def diamond():
    """seed -> a -> target and seed -> b -> target, plus a decoy branch."""
    return build_graph(
        [
            ("seed", "cites", "a"),
            ("seed", "cites", "b"),
            ("a", "cites", "target"),
            ("b", "cites", "target"),
            ("seed", "cites", "decoy"),
        ]
    )


# --------------------------------------------------------------- exactness


def test_scores_form_a_distribution(diamond):
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=200)

    assert result.converged
    assert result.scores.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(result.scores >= 0)


def test_recorded_flow_exactly_decomposes_every_score(diamond):
    """x[v] == teleport[v] + sum of recorded inflow[v], for every node."""
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=200)

    violation = result.check_decomposition(diamond)

    assert violation < 1e-12, f"decomposition off by {violation}"


def test_fast_path_matches_independent_reference_implementation(diamond):
    """The vectorised recorder and a naive per-arc loop must agree."""
    alpha = 0.15
    fast = personalized_pagerank(diamond, ["seed"], alpha=alpha, max_iterations=400,
                                 tolerance=1e-15)
    ref_scores, ref_flow = reference_flow_accumulation(
        diamond, ["seed"], alpha=alpha, iterations=400
    )

    np.testing.assert_allclose(fast.scores, ref_scores, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast.arc_flow, ref_flow, rtol=0, atol=1e-12)


def test_dangling_node_mass_is_conserved():
    """A node with no outgoing arcs must not leak probability mass."""
    graph = build_graph([("seed", "cites", "sink")])
    # Remove the reverse arc situation by checking totals only.
    result = personalized_pagerank(graph, ["seed"], alpha=0.15, max_iterations=300)

    assert result.scores.sum() == pytest.approx(1.0, abs=1e-9)


def test_result_is_deterministic_across_runs(diamond):
    first = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=200)
    second = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=200)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.arc_flow, second.arc_flow)


# ------------------------------------------------------------- attribution


def test_attribution_credits_the_edges_on_the_real_paths(diamond):
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=300)

    contributions = attribute_edges(
        diamond, result, "target", max_depth=3, min_flow_fraction=0.001
    )
    credited = {c.edge_id for c in contributions}

    assert make_edge_id("a", "cites", "target") in credited
    assert make_edge_id("b", "cites", "target") in credited


def test_attribution_credit_never_exceeds_the_candidates_own_mass(diamond):
    """Credit is a fraction of the candidate's mass, so depth-1 credit <= 1."""
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=300)

    depth_one = [
        c
        for c in attribute_edges(diamond, result, "target", max_depth=1,
                                 min_flow_fraction=0.0, max_edges=10_000)
        if c.depth == 1
    ]

    assert sum(c.credit for c in depth_one) <= 1.0 + 1e-9


def test_depth_one_credit_equals_measured_inflow_share(diamond):
    """The headline exactness claim, stated as a number a reviewer can check."""
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=300)
    idx = diamond.index["target"]
    index = FlowIndex(diamond)

    contributions = attribute_edges(
        diamond, result, "target", max_depth=1, min_flow_fraction=0.0, max_edges=10_000
    )
    total_credit = sum(c.credit for c in contributions)

    measured_inflow = sum(
        float(result.arc_flow[int(a)]) for a in index.in_arcs(idx)
    )
    expected = measured_inflow / float(result.scores[idx])

    assert total_credit == pytest.approx(expected, abs=1e-12)


def test_attribution_is_ordered_deterministically(diamond):
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=300)

    first = attribute_edges(diamond, result, "target", min_flow_fraction=0.0)
    second = attribute_edges(diamond, result, "target", min_flow_fraction=0.0)

    assert [c.edge_id for c in first] == [c.edge_id for c in second]


def test_unknown_candidate_returns_no_contributions(diamond):
    result = personalized_pagerank(diamond, ["seed"], alpha=0.15)

    assert attribute_edges(diamond, result, "not-in-graph") == []


# ---------------------------------------------------------------- ablation


def test_removing_an_edge_removes_both_arc_directions(diamond):
    edge_id = make_edge_id("a", "cites", "target")

    ablated = diamond.without_edges([edge_id])

    assert edge_id not in ablated.arc_edge_id
    assert ablated.num_arcs == diamond.num_arcs - 2


def test_ablating_a_contributing_edge_lowers_the_candidate_score(diamond):
    baseline = personalized_pagerank(diamond, ["seed"], alpha=0.15, max_iterations=300)
    edge_id = make_edge_id("a", "cites", "target")

    ablated_graph = diamond.without_edges([edge_id])
    ablated = personalized_pagerank(ablated_graph, ["seed"], alpha=0.15, max_iterations=300)

    assert ablated.score_of(ablated_graph, "target") < baseline.score_of(diamond, "target")


def test_only_edges_keeps_exactly_the_named_edges(diamond):
    keep = {make_edge_id("seed", "cites", "a"), make_edge_id("a", "cites", "target")}

    kept = diamond.only_edges(keep)

    assert kept.edge_ids == keep
    assert kept.num_arcs == 4  # two stored edges, both directions


def test_ablated_graph_still_decomposes_exactly(diamond):
    """Ablation must not break the invariant the attribution depends on."""
    ablated_graph = diamond.without_edges([make_edge_id("a", "cites", "target")])

    result = personalized_pagerank(ablated_graph, ["seed"], alpha=0.15, max_iterations=300)

    assert result.check_decomposition(ablated_graph) < 1e-12


def test_empty_graph_does_not_crash():
    graph = TypedGraph.from_rows([], [])

    result = personalized_pagerank(graph, ["seed"])

    assert result.scores.shape == (0,)
