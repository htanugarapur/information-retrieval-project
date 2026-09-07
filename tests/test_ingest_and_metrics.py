"""Tests for record normalisation, corpus loaders, metrics, and artifacts.

Normalisation is where a silent field-name change becomes a corrupt corpus, so
the shapes both APIs actually return (verified live 2026-08-10) are pinned here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from synapse.artifacts import corpus_fingerprint, update_manifest, write_artifact
from synapse.db import Database, make_edge_id
from synapse.ingest import (
    ConceptExtractor,
    _iter_json_records,
    author_node_id,
    concept_node_id,
    extract_concepts,
    load_dblp,
    load_s2orc,
    slugify,
    venue_node_id,
)
from synapse.metrics import (
    bootstrap_distribution,
    mean_reciprocal_rank,
    min_achievable_p_value,
    monte_carlo_p_value,
    ndcg_at_k,
    paired_permutation_test,
    reciprocal_rank,
)
from synapse.openalex import normalise_work, reconstruct_abstract, short_id
from synapse.s2 import normalise_paper


# ------------------------------------------------------------- identifiers


@pytest.mark.parametrize(
    "text,expected",
    [("Hello World", "hello-world"), ("Café Über", "cafe-uber"),
     ("A/B & C", "a-b-c"), ("   spaced   out  ", "spaced-out")],
)
def test_slugify_is_stable_and_ascii(text, expected):
    assert slugify(text) == expected


def test_node_id_helpers_are_prefixed():
    assert author_node_id("A1").startswith("author:")
    assert venue_node_id("SIGIR").startswith("venue:")
    assert concept_node_id("graphs").startswith("concept:")


def test_edge_id_is_derived_from_the_triple_not_insertion_order():
    assert make_edge_id("a", "cites", "b") == make_edge_id("a", "cites", "b")
    assert make_edge_id("a", "cites", "b") != make_edge_id("b", "cites", "a")


# -------------------------------------------------------------- OpenAlex


def test_short_id_strips_the_url_prefix():
    assert short_id("https://openalex.org/W123") == "W123"
    assert short_id(None) is None


def test_reconstruct_abstract_restores_token_order():
    inverted = {"dense": [0], "passage": [1], "retrieval": [2]}

    assert reconstruct_abstract(inverted) == "dense passage retrieval"


def test_reconstruct_abstract_handles_repeated_tokens():
    inverted = {"the": [0, 2], "cat": [1], "hat": [3]}

    assert reconstruct_abstract(inverted) == "the cat the hat"


def test_reconstruct_abstract_of_nothing_is_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_normalise_work_prefers_display_name_when_title_is_null():
    """OpenAlex frequently returns title=null; display_name carries it."""
    raw = {"id": "https://openalex.org/W1", "title": None,
           "display_name": "Real Title", "publication_year": 2020}

    assert normalise_work(raw)["title"] == "Real Title"


def test_normalise_work_extracts_venue_authors_topics_and_references():
    raw = {
        "id": "https://openalex.org/W1",
        "display_name": "T",
        "publication_year": 2020,
        "cited_by_count": 7,
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Ada"}}],
        "primary_location": {"source": {"display_name": "SIGIR"}},
        "topics": [{"display_name": "Information Retrieval"}],
    }

    paper = normalise_work(raw)

    assert paper["venue"] == "SIGIR"
    assert paper["authors"] == [{"author_id": "A1", "name": "Ada"}]
    assert paper["fields"] == ["Information Retrieval"]
    assert paper["references"] == ["W2", "W3"]
    assert paper["reference_count"] == 2
    assert paper["source"] == "openalex"


def test_normalise_work_rejects_a_record_without_an_id():
    assert normalise_work({"display_name": "no id"}) is None


# ---------------------------------------------------------- Semantic Scholar


def test_normalise_paper_flattens_s2_fields_of_study():
    raw = {"paperId": "abc", "title": "T",
           "s2FieldsOfStudy": [{"category": "Computer Science"},
                               {"category": "Computer Science"}]}

    assert normalise_paper(raw)["fields"] == ["Computer Science"]


def test_normalise_paper_falls_back_to_publication_venue_name():
    raw = {"paperId": "abc", "title": "T", "venue": "",
           "publicationVenue": {"name": "JCDL"}}

    assert normalise_paper(raw)["venue"] == "JCDL"


def test_normalise_paper_rejects_a_record_without_an_id():
    assert normalise_paper({"title": "orphan"}) is None


def test_normalise_paper_skips_authors_without_ids():
    raw = {"paperId": "a", "title": "T",
           "authors": [{"name": "No Id"}, {"authorId": "A2", "name": "Has Id"}]}

    assert normalise_paper(raw)["authors"] == [{"author_id": "A2", "name": "Has Id"}]


# ---------------------------------------------------------------- loaders


def test_iter_json_records_reads_jsonl(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    assert [r["id"] for r in _iter_json_records(path)] == [1, 2]


def test_iter_json_records_reads_a_json_array(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('[{"id": 1}, {"id": 2}]', encoding="utf-8")

    assert [r["id"] for r in _iter_json_records(path)] == [1, 2]


def test_iter_json_records_skips_malformed_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": 1}\nNOT JSON\n{"id": 2}\n', encoding="utf-8")

    assert [r["id"] for r in _iter_json_records(path)] == [1, 2]


def test_missing_corpus_file_names_the_documentation(tmp_path):
    with pytest.raises(FileNotFoundError, match="CORPORA.md"):
        list(_iter_json_records(tmp_path / "nope.jsonl"))


def test_load_s2orc_adds_only_internal_citation_edges(tmp_path):
    """A citation to a paper outside the subset must not create a phantom node."""
    path = tmp_path / "s2orc.jsonl"
    path.write_text(
        json.dumps({"corpusid": 1, "title": "A", "abstract": "x", "year": 2019,
                    "venue": "V", "authors": [], "references": [2, 9999]}) + "\n"
        + json.dumps({"corpusid": 2, "title": "B", "abstract": "y", "year": 2018,
                      "venue": "V", "authors": [], "references": []}) + "\n",
        encoding="utf-8",
    )
    db = Database(tmp_path / "db.sqlite")

    load_s2orc(db, path)

    assert db.count_nodes("paper") == 2
    edges = [e for e in db.all_edges() if e["rel"] == "cites"]
    assert len(edges) == 1
    assert edges[0]["dst"] == "s2orc:2"
    db.close()


def test_load_dblp_reads_venue_objects(tmp_path):
    path = tmp_path / "dblp.json"
    path.write_text(json.dumps([
        {"id": "d1", "title": "A", "abstract": "x", "year": 2019,
         "venue": {"raw": "SIGIR"}, "authors": [{"id": "a1", "name": "Ada"}],
         "references": ["d2"], "n_citation": 4},
        {"id": "d2", "title": "B", "abstract": "y", "year": 2017,
         "venue": "JCDL", "authors": [], "references": []},
    ]), encoding="utf-8")
    db = Database(tmp_path / "db.sqlite")

    load_dblp(db, path)

    assert db.get_paper("dblp:d1")["venue"] == "SIGIR"
    assert len([e for e in db.all_edges() if e["rel"] == "cites"]) == 1
    db.close()


def test_loader_respects_max_papers(tmp_path):
    path = tmp_path / "s2orc.jsonl"
    path.write_text("".join(
        json.dumps({"corpusid": i, "title": f"T{i}", "abstract": "x",
                    "year": 2019, "venue": "V", "authors": []}) + "\n"
        for i in range(10)
    ), encoding="utf-8")
    db = Database(tmp_path / "db.sqlite")

    stats = load_s2orc(db, path, max_papers=4)

    assert db.count_nodes("paper") == 4
    assert stats.frontier_truncated
    db.close()


# --------------------------------------------------------------- concepts


def test_concept_extraction_drops_singleton_concepts(corpus):
    """A concept on one paper cannot lie on a path and only inflates the graph."""
    result = extract_concepts(corpus, top_k=5, min_papers_per_concept=2, progress=False)

    assert result["concepts_dropped_singleton"] > 0
    for node in corpus.nodes_by_kind("concept"):
        holders = [
            e for e in corpus.all_edges()
            if e["rel"] == "shares_concept" and e["dst"] == node["node_id"]
        ]
        assert len(holders) >= 2


def test_concept_extractor_is_deterministic():
    extractor = ConceptExtractor(top_k=5)
    text = "Dense passage retrieval for open domain question answering with dual encoders."

    assert extractor.keyphrases(text) == extractor.keyphrases(text)


def test_concept_extractor_ignores_trivial_text():
    assert ConceptExtractor().keyphrases("hi") == []


# ---------------------------------------------------------------- metrics


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, k=2) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert ndcg_at_k(["x"], {}, k=10) == 0.0


def test_ndcg_penalises_a_reversed_ranking():
    good = ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, k=2)
    bad = ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, k=2)

    assert bad < good


def test_reciprocal_rank_finds_the_first_hit():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_with_no_hit():
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_mean_reciprocal_rank_averages():
    value = mean_reciprocal_rank([["a"], ["x", "b"]], [["a"], ["b"]])

    assert value == pytest.approx((1.0 + 0.5) / 2)


def test_bootstrap_interval_brackets_the_mean():
    d = bootstrap_distribution([1, 2, 3, 4, 5], samples=500, seed=1)

    assert d.ci_low <= d.mean <= d.ci_high
    assert d.n == 5


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    a = bootstrap_distribution([1, 5, 2, 8], samples=400, seed=7)
    b = bootstrap_distribution([1, 5, 2, 8], samples=400, seed=7)

    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_bootstrap_of_nothing_is_empty_not_a_crash():
    assert bootstrap_distribution([]).n == 0


def test_bootstrap_of_a_single_value_has_a_degenerate_interval():
    d = bootstrap_distribution([3.0])

    assert d.mean == d.ci_low == d.ci_high == 3.0


def test_distribution_format_always_shows_the_interval():
    text = bootstrap_distribution([1, 2, 3], samples=200, seed=0).format()

    assert "[" in text and "]" in text and "n=3" in text


def test_paired_test_detects_a_real_difference():
    treatment = [10, 12, 9, 11, 13, 10, 12]
    control = [1, 2, 1, 0, 2, 1, 1]

    result = paired_permutation_test(treatment, control, samples=2000, seed=0)

    assert result.significant and result.p_value < 0.05


def test_paired_test_finds_nothing_in_identical_samples():
    result = paired_permutation_test([1, 2, 3], [1, 2, 3], samples=500, seed=0)

    assert result.p_value == 1.0 and not result.significant


def test_paired_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_permutation_test([1, 2], [1], samples=10)


def test_monte_carlo_p_value_never_returns_zero():
    """A finite resample cannot justify p = 0."""
    assert monte_carlo_p_value(100.0, [1.0] * 30) > 0


def test_monte_carlo_p_value_is_one_when_control_always_wins():
    assert monte_carlo_p_value(0.0, [5.0] * 10) == pytest.approx(1.0)


def test_min_achievable_p_value_matches_the_trial_count():
    assert min_achievable_p_value(30) == pytest.approx(1 / 31)


# -------------------------------------------------------------- artifacts


def test_corpus_fingerprint_changes_when_an_edge_is_added(corpus):
    before = corpus_fingerprint(corpus)["edge_digest"]
    corpus.add_edge("P_SEED", "cites", "P_UNFAITH")

    assert corpus_fingerprint(corpus)["edge_digest"] != before


def test_write_artifact_stamps_provenance(tmp_path, corpus, config):
    path = write_artifact({"x": 1}, tmp_path / "a.json", config.to_dict(), corpus)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["provenance"]["corpus"]["papers"] > 0
    assert "config" in payload["provenance"]


def test_write_artifact_serialises_numpy_scalars(tmp_path):
    path = write_artifact({"value": np.float64(1.5)}, tmp_path / "a.json")

    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1.5


def test_write_artifact_refuses_an_unserialisable_object(tmp_path):
    class Opaque:
        pass

    with pytest.raises(TypeError, match="cannot serialise"):
        write_artifact({"x": Opaque()}, tmp_path / "a.json")


def test_manifest_lists_trees_and_drops_deleted_ones(tmp_path):
    (tmp_path / "tree_S1.json").write_text(json.dumps(
        {"seed": "S1", "nodes": [{"id": "S1", "kind": "seed", "title": "T"}], "tiers": ["a"]}
    ), encoding="utf-8")

    manifest = json.loads(update_manifest(tmp_path).read_text(encoding="utf-8"))
    assert [t["seed"] for t in manifest["trees"]] == ["S1"]

    (tmp_path / "tree_S1.json").unlink()
    manifest = json.loads(update_manifest(tmp_path).read_text(encoding="utf-8"))
    assert manifest["trees"] == []


def test_manifest_survives_a_corrupt_artifact(tmp_path):
    (tmp_path / "tree_bad.json").write_text("{not json", encoding="utf-8")

    manifest = json.loads(update_manifest(tmp_path).read_text(encoding="utf-8"))

    assert manifest["trees"] == []
