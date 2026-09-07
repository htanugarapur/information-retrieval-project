"""Shared fixtures: a synthetic corpus with known causal structure.

This corpus exists so the ablation harness can be tested against ground truth we
control. It is NEVER a source of paper numbers -- runs built on it are tagged
`synthetic` in artifacts. Its job is to contain, by construction:

  * a candidate whose rank genuinely depends on a citation path (faithful)
  * a candidate that ranks on text alone while sharing only a huge venue with
    the seed (unfaithful -- deleting the venue edge should barely move it)

If the harness cannot tell those two apart, it does not work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synapse.config import Config
from synapse.db import Database
from synapse.graph import TypedGraph
from synapse.ingest import author_node_id, venue_node_id, concept_node_id


TEST_CONFIG = {
    "seed": 12345,
    "paths": {"db": "data/test.sqlite", "runs": "runs"},
    "retrieval": {
        "bm25": {"k1": 1.5, "b": 0.75},
        "dense": {"model": "sentence-transformers/all-MiniLM-L6-v2", "batch_size": 8},
        "rrf": {"k": 60},
        "ppr": {
            "alpha": 0.15,
            "max_iterations": 100,
            "tolerance": 1e-12,
            "edge_weights": {
                "cites": 1.0,
                "cited_by": 1.0,
                "authored_by": 0.6,
                "published_in": 0.3,
                "shares_concept": 0.5,
            },
            "attribution": {
                "max_depth": 3,
                "min_flow_fraction": 0.001,
                "max_edges_per_candidate": 40,
            },
        },
        "fusion": {"rrf_weight": 0.6, "ppr_weight": 0.4},
        "candidate_pool": 100,
        "top_k": 10,
    },
    "ablate": {
        "random_control_trials": 30,
        "bootstrap_samples": 300,
        "confidence_level": 0.95,
        "significance_alpha": 0.05,
        "sufficiency": True,
    },
    "tree": {
        "max_nodes": 15,
        "max_tiers": 6,
        "citation_dominance_ratio": 3.0,
    },
}


@pytest.fixture
def config() -> Config:
    return Config(TEST_CONFIG)


def _paper(pid, title, abstract, year, venue, cites=0, authors=(), fields=()):
    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "citation_count": cites,
        "reference_count": 0,
        "fields": list(fields),
        "authors": [{"author_id": a, "name": a.title()} for a in authors],
        "source": "synthetic",
        "hop": 0,
    }


@pytest.fixture
def corpus(tmp_path) -> Database:
    """A 14-paper synthetic corpus with hand-placed causal structure."""
    db = Database(tmp_path / "corpus.sqlite")

    papers = [
        # The seed: graph-based retrieval.
        _paper("P_SEED", "Graph based retrieval of scientific papers",
               "We rank scientific papers using a citation graph and personalized pagerank "
               "over typed edges between papers authors and venues.",
               2021, "SIGIR", 40, ("alice", "bob"), ("Computer Science",)),

        # FAITHFUL case: connected to the seed by citation and shared author,
        # with weak lexical overlap so the graph is doing the work.
        _paper("P_FAITH", "Random walks with restart on bibliographic networks",
               "A method for propagating relevance through heterogeneous bibliographic "
               "networks using restart probability and typed adjacency.",
               2019, "JCDL", 120, ("alice",), ("Computer Science",)),

        # UNFAITHFUL case: strong lexical overlap with the seed, but its ONLY
        # graph connection is a venue shared with hundreds of papers.
        _paper("P_UNFAITH", "Graph based retrieval of scientific papers revisited",
               "We rank scientific papers using a citation graph and personalized pagerank "
               "over typed edges between papers authors and venues, with new experiments.",
               2022, "SIGIR", 5, ("zoe",), ("Computer Science",)),

        _paper("P_REF1", "Personalized pagerank computation",
               "Efficient algorithms for computing personalized pagerank vectors on large graphs.",
               2016, "WWW", 300, ("carol",), ("Computer Science",)),
        _paper("P_REF2", "Typed heterogeneous information networks",
               "Meta path based similarity search in heterogeneous information networks.",
               2015, "VLDB", 500, ("dave",), ("Computer Science",)),
        _paper("P_CITER", "A survey of scholarly recommendation",
               "We survey scholarly paper recommendation systems including graph and text methods.",
               2023, "TOIS", 20, ("erin",), ("Computer Science",)),
    ]
    # Filler papers sharing the SIGIR venue, so the venue edge is genuinely
    # low-information -- which is what makes P_UNFAITH's explanation unfaithful.
    for i in range(8):
        papers.append(
            _paper(f"P_FILL{i}", f"Unrelated retrieval study number {i}",
                   "An unrelated study about query logs and click models in web search.",
                   2018 + (i % 4), "SIGIR", 10 + i, (f"filler{i}",), ("Computer Science",))
        )

    for paper in papers:
        db.upsert_paper(paper)
        for author in paper["authors"]:
            node_id = author_node_id(author["author_id"])
            db.upsert_node(node_id, "author", author["name"])
            db.add_edge(paper["paper_id"], "authored_by", node_id)
        if paper["venue"]:
            node_id = venue_node_id(paper["venue"])
            db.upsert_node(node_id, "venue", paper["venue"])
            db.add_edge(paper["paper_id"], "published_in", node_id)

    # Citation structure. P_FAITH is wired to the seed; P_UNFAITH is not.
    for src, dst in [
        ("P_SEED", "P_FAITH"),
        ("P_SEED", "P_REF1"),
        ("P_SEED", "P_REF2"),
        ("P_FAITH", "P_REF1"),
        ("P_FAITH", "P_REF2"),
        ("P_CITER", "P_SEED"),
        ("P_CITER", "P_FAITH"),
    ]:
        db.add_edge(src, "cites", dst)

    # Concepts: one shared between seed and P_FAITH, one broad one.
    db.upsert_node(concept_node_id("citation graph"), "concept", "citation graph")
    db.add_edge("P_SEED", "shares_concept", concept_node_id("citation graph"), 0.9)
    db.add_edge("P_FAITH", "shares_concept", concept_node_id("citation graph"), 0.8)

    db.upsert_node(concept_node_id("retrieval"), "concept", "retrieval")
    for pid in ["P_SEED", "P_UNFAITH", "P_CITER"] + [f"P_FILL{i}" for i in range(8)]:
        db.add_edge(pid, "shares_concept", concept_node_id("retrieval"), 0.4)

    yield db
    db.close()


@pytest.fixture
def graph(corpus, config) -> TypedGraph:
    return TypedGraph.from_db(
        corpus, edge_weights=config.get_path("retrieval.ppr.edge_weights")
    )
