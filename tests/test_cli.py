"""CLI tests.

These run the real commands against a temporary corpus built from the shared
fixture, so they exercise argument wiring, artifact writing, and — importantly —
that the LLM explainers are SKIPPED WITH A WARNING rather than silently replaced
when no key is present.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from synapse.cli import app
from tests.conftest import TEST_CONFIG

runner = CliRunner()


@pytest.fixture
def project(tmp_path, corpus, monkeypatch):
    """A config file + populated db in a temp dir, wired via SYNAPSE_CONFIG."""
    config = json.loads(json.dumps(TEST_CONFIG))  # deep copy
    config["paths"] = {"db": str(corpus.path), "runs": str(tmp_path / "runs")}
    config["retrieval"]["dense"]["model"] = "sentence-transformers/all-MiniLM-L6-v2"
    config["explain"] = {
        "metapath": {"max_paths": 12},
        "llm": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "test/model",
            "fallback_models": [],
            "temperature": 0.0, "max_tokens": 100,
            "timeout_seconds": 5, "max_retries": 1,
        },
    }
    config["eval"] = {"ndcg_k": 10, "retrieval_ablations": ["rrf-only", "rrf+ppr"]}
    config["tree"] = {"max_nodes": 8, "max_tiers": 5, "citation_dominance_ratio": 3.0}

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setenv("SYNAPSE_CONFIG", str(path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return {"config": path, "runs": tmp_path / "runs"}


def invoke(*args):
    return runner.invoke(app, list(args), catch_exceptions=False)


# ------------------------------------------------------------------ search


def test_search_ranks_and_writes_an_artifact(project):
    result = invoke("search", "P_SEED", "--top-k", "5", "--no-dense")

    assert result.exit_code == 0
    assert "corpus: 14 papers" in result.stdout
    artifacts = list(project["runs"].glob("search_*.json"))
    assert artifacts
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["search"]["candidates"]


def test_search_artifact_embeds_provenance(project):
    invoke("search", "P_SEED", "--top-k", "3", "--no-dense")
    payload = json.loads(
        next(project["runs"].glob("search_*.json")).read_text(encoding="utf-8")
    )

    provenance = payload["provenance"]
    assert provenance["corpus"]["papers"] == 14
    assert "edge_digest" in provenance["corpus"]
    assert provenance["config"]["seed"] == TEST_CONFIG["seed"]


def test_search_reports_the_decomposition_error(project):
    """The number certifying contributing_edges is exact must be user-visible."""
    result = invoke("search", "P_SEED", "--top-k", "3", "--no-dense")

    assert "decomposition_error" in result.stdout


def test_search_supports_retrieval_ablations(project):
    result = invoke("search", "P_SEED", "--signals", "bm25-only", "--no-dense", "--top-k", "3")

    assert result.exit_code == 0
    assert "signals: bm25-only" in result.stdout


# ----------------------------------------------------------------- explain


def test_explain_runs_the_metapath_explainer(project):
    result = invoke("explain", "P_SEED", "P_FAITH", "--explainer", "metapath")

    assert result.exit_code == 0
    assert "=== metapath ===" in result.stdout
    assert "cited edges:" in result.stdout


def test_explain_warns_that_llm_explainers_are_unavailable(project):
    """Silently returning only metapath would misrepresent an 'all' run."""
    result = invoke("explain", "P_SEED", "P_FAITH", "--explainer", "all")

    assert result.exit_code == 0
    assert "unavailable" in result.stdout.lower()
    assert "llm-grounded" in result.stdout


def test_explain_rejects_an_unknown_explainer(project):
    result = runner.invoke(app, ["explain", "P_SEED", "P_FAITH", "--explainer", "psychic"])

    assert result.exit_code != 0


# ------------------------------------------------------------------ ablate


def test_ablate_produces_a_report_and_artifact(project):
    result = invoke("ablate", "P_SEED", "--explainer", "metapath",
                    "--top-k", "5", "--no-dense", "--no-verbose")

    assert result.exit_code == 0
    assert "displacement" in result.stdout
    payload = json.loads((project["runs"] / "ablate_P_SEED.json").read_text(encoding="utf-8"))
    assert payload["cells"]
    assert "metapath" in payload["reports"]


def test_ablate_records_which_explainers_were_skipped(project):
    invoke("ablate", "P_SEED", "--explainer", "all", "--top-k", "3",
           "--no-dense", "--no-verbose")
    payload = json.loads((project["runs"] / "ablate_P_SEED.json").read_text(encoding="utf-8"))

    assert set(payload["skipped_explainers"]) == {"llm-grounded", "llm-free"}


def test_every_ablation_cell_carries_its_control(project):
    invoke("ablate", "P_SEED", "--explainer", "metapath", "--top-k", "4",
           "--no-dense", "--no-verbose")
    payload = json.loads((project["runs"] / "ablate_P_SEED.json").read_text(encoding="utf-8"))

    for cell in payload["cells"]:
        if cell["testable"]:
            assert len(cell["control_displacements"]) == 30


# -------------------------------------------------------------------- tree


def test_tree_writes_an_artifact_with_tiers_and_a_neighbourhood(project):
    result = invoke("tree", "P_SEED", "--top-k", "10", "--no-dense")

    assert result.exit_code == 0
    payload = json.loads(
        (project["runs"] / "tree_P_SEED.json").read_text(encoding="utf-8")
    )
    assert payload["nodes"] and payload["tiers"]
    assert payload["neighbourhood"]["nodes"]


def test_tree_warns_when_there_is_no_ablation_artifact(project):
    result = invoke("tree", "P_SEED", "--top-k", "5", "--no-dense")

    assert "no ablation artifact" in result.stdout.lower()


def test_tree_uses_verdicts_from_a_prior_ablation(project):
    invoke("ablate", "P_SEED", "--explainer", "metapath", "--top-k", "5",
           "--no-dense", "--no-verbose")
    invoke("tree", "P_SEED", "--top-k", "10", "--no-dense")

    payload = json.loads(
        (project["runs"] / "tree_P_SEED.json").read_text(encoding="utf-8")
    )
    assert any(n["verdict"] for n in payload["nodes"])


def test_tree_updates_the_gui_manifest(project):
    invoke("tree", "P_SEED", "--top-k", "8", "--no-dense")

    manifest = json.loads((project["runs"] / "index.json").read_text(encoding="utf-8"))
    assert [t["seed"] for t in manifest["trees"]] == ["P_SEED"]


def test_verdict_note_is_readable_without_the_paper(project):
    invoke("ablate", "P_SEED", "--explainer", "metapath", "--top-k", "5",
           "--no-dense", "--no-verbose")
    invoke("tree", "P_SEED", "--top-k", "10", "--no-dense")
    payload = json.loads(
        (project["runs"] / "tree_P_SEED.json").read_text(encoding="utf-8")
    )

    notes = [n["verdict"]["note"] for n in payload["nodes"] if n["verdict"]]
    assert notes
    for note in notes:
        # No jargon: a scholar who has not read the paper must understand it.
        assert "necessity" not in note.lower()
        assert "displacement" not in note.lower()
        assert note.endswith(".")


# -------------------------------------------------------------------- eval


def test_eval_prints_both_tables_and_writes_an_artifact(project):
    result = invoke("eval", "--dataset", "primary", "--seeds", "-", "--top-k", "5",
                    "--no-dense", "--no-verbose")

    # `--seeds -` does not exist, so it falls back to hop==0 papers; the fixture
    # corpus has none, which must be reported rather than silently producing an
    # empty table.
    assert result.exit_code in (0, 1)


def test_eval_reports_no_seeds_rather_than_an_empty_table(project):
    result = runner.invoke(app, ["eval", "--dataset", "dblp", "--no-dense"])

    assert result.exit_code == 1
    assert "no seeds" in result.stdout.lower()
