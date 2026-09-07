"""Tests for the ablation protocol — the paper's contribution.

The decisive test is `test_harness_separates_the_faithful_from_the_unfaithful`:
the fixture corpus contains one candidate whose rank is genuinely caused by
citation structure and one that ranks on text while sharing only a crowded
venue. A harness that scores those two the same is measuring nothing.
"""

from __future__ import annotations

import random

import pytest

from synapse.ablate import (
    AblationHarness,
    necessity_score,
    sample_control_edges,
)
from synapse.explain import MetaPathExplainer, MetaPathFinder
from synapse.metrics import min_achievable_p_value
from synapse.retrieve import Retriever


@pytest.fixture
def retriever(corpus, config):
    # Sparse-only keeps the suite fast and CPU-light; the graph mechanics under
    # test are identical either way, and the dense path has its own test.
    return Retriever(corpus, config, load_dense=False)


@pytest.fixture
def harness(retriever, config):
    return AblationHarness(retriever, config)


@pytest.fixture
def explainer(graph, corpus):
    return MetaPathExplainer(MetaPathFinder(graph, corpus))


# ------------------------------------------------------ necessity scoring


def test_necessity_is_one_when_candidate_falls_to_the_bottom():
    assert necessity_score(baseline_rank=1, ablated_rank=100, list_length=100) == 1.0


def test_necessity_is_zero_when_nothing_moves():
    assert necessity_score(baseline_rank=5, ablated_rank=5, list_length=100) == 0.0


def test_necessity_normalises_across_different_list_positions():
    """Same proportional fall from different starting ranks scores the same."""
    top = necessity_score(baseline_rank=1, ablated_rank=51, list_length=101)
    lower = necessity_score(baseline_rank=51, ablated_rank=76, list_length=101)

    assert top == pytest.approx(0.5)
    assert lower == pytest.approx(0.5)


def test_negative_necessity_is_preserved_not_clipped():
    """An explanation whose edges were holding a paper DOWN is a real result."""
    assert necessity_score(baseline_rank=10, ablated_rank=5, list_length=100) < 0


def test_necessity_handles_last_place_without_dividing_by_zero():
    assert necessity_score(baseline_rank=100, ablated_rank=100, list_length=100) == 0.0


# --------------------------------------------------------- random control


def test_control_pool_keeps_the_cited_edges_by_default(graph):
    """Exchangeability null: the cited edges must remain drawable."""
    cited = graph.incident_edge_ids("P_SEED")

    draws = set()
    for trial in range(200):
        sample = sample_control_edges(
            graph, "P_SEED", "P_FAITH", 2, exclude=cited, rng=random.Random(trial)
        )
        draws.update(sample or [])

    assert draws & set(cited), "cited edges were excluded from the control pool"


def test_disjoint_mode_excludes_cited_edges_when_explicitly_requested(graph):
    rng = random.Random(0)
    cited = graph.incident_edge_ids("P_SEED")[:2]

    sample = sample_control_edges(
        graph, "P_SEED", "P_FAITH", 2, exclude=cited, rng=rng, disjoint=True
    )

    assert sample is not None
    assert not set(sample) & set(cited)


def test_control_matches_the_requested_edge_count(graph):
    rng = random.Random(0)

    sample = sample_control_edges(graph, "P_SEED", "P_FAITH", 3, exclude=[], rng=rng)

    assert sample is not None and len(sample) == 3


def test_control_returns_none_when_pool_is_too_small(graph):
    """Better to mark a case untestable than to compare unequal interventions."""
    rng = random.Random(0)

    sample = sample_control_edges(graph, "P_SEED", "P_FAITH", 10_000, exclude=[], rng=rng)

    assert sample is None


def test_control_sampling_is_deterministic_for_a_fixed_seed(graph):
    first = sample_control_edges(graph, "P_SEED", "P_FAITH", 3, [], random.Random(7))
    second = sample_control_edges(graph, "P_SEED", "P_FAITH", 3, [], random.Random(7))

    assert first == second


# ----------------------------------------------------------- harness guard


def test_harness_refuses_a_trial_count_that_can_never_reach_significance(retriever, config):
    from synapse.config import Config

    broken = Config({**config.to_dict(), "ablate": {**config["ablate"], "random_control_trials": 5}})

    with pytest.raises(ValueError, match="minimum"):
        AblationHarness(retriever, broken)


def test_min_achievable_p_matches_the_documented_floor():
    assert min_achievable_p_value(30) == pytest.approx(1 / 31)


# ------------------------------------------------------------- the protocol


def test_ablation_records_baseline_and_ablated_ranks(harness, explainer, retriever):
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)
    explanation = explainer.explain("P_SEED", "P_FAITH")

    cell = harness.ablate_candidate("P_SEED", "P_FAITH", explanation, baseline_ranks)

    assert cell.baseline_rank >= 1
    assert cell.ablated_rank >= 1
    assert cell.displacement == cell.ablated_rank - cell.baseline_rank


def test_every_ablation_runs_its_control(harness, explainer, retriever):
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)
    explanation = explainer.explain("P_SEED", "P_FAITH")

    cell = harness.ablate_candidate("P_SEED", "P_FAITH", explanation, baseline_ranks)

    assert cell.testable
    assert len(cell.control_displacements) == harness.control_trials


def test_empty_explanation_is_untestable_not_scored_as_zero(harness, retriever):
    """Citing nothing must not be laundered into 'caused no harm'."""
    from synapse.explain import Explanation

    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)
    empty = Explanation(explainer="metapath", seed="P_SEED", candidate="P_FAITH",
                        text="nothing", cited_edges=[])

    cell = harness.ablate_candidate("P_SEED", "P_FAITH", empty, baseline_ranks)

    assert cell.testable is False
    assert "cited no edges" in cell.untestable_reason


def test_ablation_does_not_mutate_the_original_graph(harness, explainer, retriever):
    before = set(retriever.graph.edge_ids)
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)

    harness.ablate_candidate(
        "P_SEED", "P_FAITH", explainer.explain("P_SEED", "P_FAITH"), baseline_ranks
    )

    assert set(retriever.graph.edge_ids) == before


def test_sufficiency_is_computed(harness, explainer, retriever):
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)

    cell = harness.ablate_candidate(
        "P_SEED", "P_FAITH", explainer.explain("P_SEED", "P_FAITH"), baseline_ranks
    )

    assert cell.sufficiency_rank is not None
    assert cell.sufficiency_retention is not None


def test_results_are_reproducible_across_identical_runs(retriever, config, explainer):
    first = AblationHarness(retriever, config)
    second = AblationHarness(retriever, config)
    baseline = first._full_ranking("P_SEED", retriever.graph)
    explanation = explainer.explain("P_SEED", "P_FAITH")

    a = first.ablate_candidate("P_SEED", "P_FAITH", explanation, baseline)
    b = second.ablate_candidate("P_SEED", "P_FAITH", explanation, baseline)

    assert a.displacement == b.displacement
    assert a.control_displacements == b.control_displacements
    assert a.p_value == b.p_value


# ------------------------------------------------------- the decisive test


def test_harness_separates_the_faithful_from_the_unfaithful(harness, explainer, retriever):
    """The whole instrument, on ground truth we constructed.

    P_FAITH is wired to the seed by citation and a shared author.
    P_UNFAITH shares only a venue that eleven papers sit on.
    The citation-backed explanation must displace more than the venue-backed one.
    """
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)

    faithful = harness.ablate_candidate(
        "P_SEED", "P_FAITH", explainer.explain("P_SEED", "P_FAITH"), baseline_ranks
    )
    unfaithful = harness.ablate_candidate(
        "P_SEED", "P_UNFAITH", explainer.explain("P_SEED", "P_UNFAITH"), baseline_ranks
    )

    assert faithful.necessity_score > unfaithful.necessity_score, (
        f"instrument cannot distinguish causal structure from coincidence: "
        f"faithful={faithful.necessity_score:.4f} "
        f"unfaithful={unfaithful.necessity_score:.4f}"
    )


def test_summary_never_reports_a_mean_without_an_interval(harness, explainer, retriever):
    baseline_ranks = harness._full_ranking("P_SEED", retriever.graph)
    cells = [
        harness.ablate_candidate(
            "P_SEED", candidate, explainer.explain("P_SEED", candidate), baseline_ranks
        )
        for candidate in ("P_FAITH", "P_UNFAITH", "P_REF1")
    ]

    report = harness.summarise("metapath", cells)

    assert report.displacement.ci_low <= report.displacement.mean <= report.displacement.ci_high
    assert report.necessity.n == report.n_testable
    assert report.paired_test.n == report.n_testable
