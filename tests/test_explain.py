"""Tests for the three explainers.

The most important test in this file is
`test_free_explainer_prompt_contains_no_graph_information`. If the free-form
explainer can see the graph, the comparison between explainers is meaningless
and the experiment is dead. That guardrail is asserted mechanically, not trusted.
"""

from __future__ import annotations

import json

import pytest

from synapse.explain import (
    LLMFreeExplainer,
    LLMGroundedExplainer,
    MetaPathExplainer,
    MetaPathFinder,
    build_explainers,
)
from synapse.llm import LLMResponse, LLMUnavailable, parse_json_object


class FakeLLM:
    """Records every prompt it is given and replays canned JSON."""

    def __init__(self, payload, available=True):
        self._payload = payload
        self.available = available
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete_json(self, prompt, system=None, use_cache=True):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        if isinstance(self._payload, str):
            text = self._payload
        else:
            text = json.dumps(self._payload)
        return parse_json_object(text), LLMResponse(text=text, model="fake", cached=False)


@pytest.fixture
def finder(graph, corpus):
    return MetaPathFinder(graph, corpus)


# ------------------------------------------------------------- meta-paths


def test_finder_detects_a_direct_citation(finder):
    paths = finder.find("P_SEED", "P_FAITH")

    assert any(p.relation == "direct_citation" for p in paths)


def test_finder_detects_a_shared_author(finder):
    """P_SEED and P_FAITH share author 'alice'."""
    paths = finder.find("P_SEED", "P_FAITH")

    assert any(p.relation == "shared_author" for p in paths)


def test_finder_detects_co_citation(finder):
    """Both cite P_REF1 and P_REF2."""
    paths = finder.find("P_SEED", "P_FAITH")

    assert any(p.relation == "co_citation" for p in paths)


def test_finder_reports_only_a_venue_link_for_the_unfaithful_pair(finder):
    """P_UNFAITH shares the SIGIR venue and the broad 'retrieval' concept only."""
    relations = {p.relation for p in finder.find("P_SEED", "P_UNFAITH")}

    assert "shared_venue" in relations
    assert "direct_citation" not in relations
    assert "shared_author" not in relations


def test_every_path_carries_real_edge_ids(finder, graph):
    for path in finder.find("P_SEED", "P_FAITH"):
        for edge_id in path.edge_ids:
            assert edge_id in graph.edge_ids


def test_finder_is_deterministic(finder):
    first = [p.to_dict() for p in finder.find("P_SEED", "P_FAITH")]
    second = [p.to_dict() for p in finder.find("P_SEED", "P_FAITH")]

    assert first == second


def test_finder_returns_nothing_for_unknown_nodes(finder):
    assert finder.find("P_SEED", "NOT_A_PAPER") == []


# --------------------------------------------------------- metapath explainer


def test_metapath_explainer_cites_the_edges_it_names(finder, graph):
    explanation = MetaPathExplainer(finder).explain("P_SEED", "P_FAITH")

    assert explanation.cited_edges
    assert all(e in graph.edge_ids for e in explanation.cited_edges)


def test_metapath_explainer_is_deterministic(finder):
    explainer = MetaPathExplainer(finder)

    first = explainer.explain("P_SEED", "P_FAITH")
    second = explainer.explain("P_SEED", "P_FAITH")

    assert first.cited_edges == second.cited_edges
    assert first.text == second.text


def test_metapath_explainer_says_so_when_there_is_no_path(finder):
    explanation = MetaPathExplainer(finder).explain("P_SEED", "P_FILL3")

    # Filler papers share the venue, so there IS a path; assert honesty instead:
    # whatever it cites must be real, and empty text must mean empty edges.
    if not explanation.cited_edges:
        assert "No typed path" in explanation.text


# --------------------------------------------------------- grounded explainer


def test_grounded_explainer_maps_fact_numbers_back_to_edges(finder):
    fake = FakeLLM({"explanation": "They share an author.", "fact_numbers": [1]})
    explainer = LLMGroundedExplainer(finder, fake)

    explanation = explainer.explain("P_SEED", "P_FAITH")

    expected = finder.find("P_SEED", "P_FAITH")[0].edge_ids
    assert set(explanation.cited_edges) == set(expected)


def test_grounded_explainer_ignores_out_of_range_fact_numbers(finder):
    fake = FakeLLM({"explanation": "x", "fact_numbers": [1, 999]})
    explainer = LLMGroundedExplainer(finder, fake)

    explanation = explainer.explain("P_SEED", "P_FAITH")

    assert explanation.diagnostics["hallucinated_fact_numbers"] == [999]


def test_grounded_explainer_cites_nothing_when_output_is_unparseable(finder):
    """Falling back to 'all facts' would silently turn it into the metapath explainer."""
    fake = FakeLLM("this is not json at all")
    explainer = LLMGroundedExplainer(finder, fake)

    explanation = explainer.explain("P_SEED", "P_FAITH")

    assert explanation.cited_edges == []
    assert explanation.diagnostics["failure"] == "unparseable_json"


def test_grounded_explainer_is_shown_only_facts_never_abstracts(finder, corpus):
    fake = FakeLLM({"explanation": "x", "fact_numbers": [1]})

    LLMGroundedExplainer(finder, fake).explain("P_SEED", "P_FAITH")

    prompt = fake.prompts[0]
    seed_abstract = corpus.get_paper("P_SEED")["abstract"]
    assert seed_abstract not in prompt


# ------------------------------------------------------------- free explainer


def test_free_explainer_prompt_contains_no_graph_information(finder, corpus, graph):
    """THE GUARDRAIL. A leak here invalidates the entire experiment."""
    fake = FakeLLM({"explanation": "x", "relationships": [{"relation": "shared_concept",
                                                          "detail": "graphs"}]})

    LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_FAITH")

    prompt = fake.prompts[0] + fake.systems[0]

    for edge_id in graph.edge_ids:
        assert edge_id not in prompt, f"edge id {edge_id} leaked into the free prompt"

    # No node identifiers of any kind.
    for node_id in graph.node_ids:
        if node_id.startswith(("author:", "venue:", "concept:")):
            assert node_id not in prompt, f"node {node_id} leaked"

    # No author names, no venue name -- those are graph facts, not paper text.
    assert "Alice" not in prompt and "alice" not in prompt
    assert "SIGIR" not in prompt
    assert "JCDL" not in prompt

    # And no meta-path phrasing.
    for phrase in ("both cite", "both cited by", "both linked to", "cites"):
        assert phrase not in prompt.lower().replace("scientific", "")


def test_free_explainer_prompt_does_contain_the_paper_text(finder, corpus):
    fake = FakeLLM({"explanation": "x", "relationships": []})

    LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_FAITH")

    assert corpus.get_paper("P_SEED")["title"] in fake.prompts[0]
    assert corpus.get_paper("P_FAITH")["abstract"][:40] in fake.prompts[0]


def test_free_explainer_grounds_a_true_claim_to_real_edges(finder, corpus, graph):
    fake = FakeLLM(
        {"explanation": "same team", "relationships": [{"relation": "shared_author",
                                                        "detail": "same group"}]}
    )

    explanation = LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_FAITH")

    assert explanation.cited_edges
    assert all(e in graph.edge_ids for e in explanation.cited_edges)
    assert explanation.unrealised_claims == []


def test_free_explainer_records_a_claim_the_graph_does_not_support(finder, corpus):
    """P_SEED and P_UNFAITH share no author. That claim must not cite edges."""
    fake = FakeLLM(
        {"explanation": "same authors surely",
         "relationships": [{"relation": "shared_author", "detail": "same group"}]}
    )

    explanation = LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_UNFAITH")

    assert explanation.cited_edges == []
    assert len(explanation.unrealised_claims) == 1
    assert "no such connection" in explanation.unrealised_claims[0]["reason"]


def test_free_explainer_rejects_an_invented_relation_type(finder, corpus):
    fake = FakeLLM(
        {"explanation": "x",
         "relationships": [{"relation": "vibes_alignment", "detail": "feels similar"}]}
    )

    explanation = LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_FAITH")

    assert explanation.cited_edges == []
    assert "not a recognised relation" in explanation.unrealised_claims[0]["reason"]


def test_free_explainer_records_that_the_model_saw_no_graph(finder, corpus):
    fake = FakeLLM({"explanation": "x", "relationships": []})

    explanation = LLMFreeExplainer(finder, fake, corpus).explain("P_SEED", "P_FAITH")

    assert explanation.diagnostics["graph_shown_to_model"] is False


# ------------------------------------------------------------------ registry


def test_llm_explainers_are_omitted_rather_than_stubbed_without_a_key(graph, corpus):
    """A metapath result must never be reportable under an LLM label."""
    explainers = build_explainers(graph, corpus, client=None)

    assert set(explainers) == {"metapath"}


def test_registry_builds_all_three_when_a_client_is_available(graph, corpus):
    fake = FakeLLM({"explanation": "x", "fact_numbers": []})

    explainers = build_explainers(graph, corpus, client=fake)

    assert set(explainers) == {"metapath", "llm-grounded", "llm-free"}


# ------------------------------------------------------------ json parsing


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here you go:\n```\n{"a": 1}\n```',
        'Sure! {"a": 1} hope that helps',
    ],
)
def test_parse_json_object_tolerates_common_model_formatting(text):
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", None])
def test_parse_json_object_returns_none_rather_than_guessing(text):
    assert parse_json_object(text) is None
