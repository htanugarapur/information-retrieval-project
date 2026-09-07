"""Phase 5 — the CLI and experiment runner.

    synapse build   --seeds ids.txt --hops 2
    synapse search  "<query or paper id>" --top-k 20
    synapse explain <seed> <candidate> --explainer metapath|llm-grounded|llm-free
    synapse ablate  <seed> --explainer all --output runs/exp1.json
    synapse tree    <seed> --output runs/tree1.json
    synapse eval    --dataset s2orc|dblp --report
    synapse serve

Every command that produces results writes a complete artifact to runs/.
"""

from __future__ import annotations

import json
import logging
import posixpath
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import typer

from .ablate import AblationHarness
from .artifacts import provenance, update_manifest, write_artifact
from .config import load_config, load_credentials
from .db import Database
from .explain import build_explainers
from .llm import OpenRouterClient
from .metrics import bootstrap_distribution, ndcg_at_k, reciprocal_rank
from .retrieve import Retriever
from .tree import build_tree, write_tree

app = typer.Typer(
    add_completion=False,
    help="Synapse — measuring whether retrieval explanations are causal.",
)

EXPLAINER_NAMES = ("metapath", "llm-grounded", "llm-free")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _context(config_path: str | None):
    config = load_config(config_path)
    db = Database(config.resolve("paths.db"))
    return config, db


def _llm_client(config, db) -> OpenRouterClient:
    credentials = load_credentials()
    return OpenRouterClient(
        db,
        api_key=credentials.openrouter_api_key,
        base_url=str(config.get_path("explain.llm.base_url")),
        model=str(config.get_path("explain.llm.model")),
        fallback_models=config.get_path("explain.llm.fallback_models") or [],
        temperature=float(config.get_path("explain.llm.temperature", 0.0)),
        max_tokens=int(config.get_path("explain.llm.max_tokens", 900)),
        timeout_seconds=int(config.get_path("explain.llm.timeout_seconds", 120)),
        max_retries=int(config.get_path("explain.llm.max_retries", 4)),
        seed=int(config.require("seed")),
    )


def _resolve_explainers(requested: str) -> list[str]:
    if requested == "all":
        return list(EXPLAINER_NAMES)
    names = [n.strip() for n in requested.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPLAINER_NAMES]
    if unknown:
        raise typer.BadParameter(
            f"unknown explainer(s): {', '.join(unknown)}. "
            f"choose from: {', '.join(EXPLAINER_NAMES)}, or 'all'"
        )
    return names


# ------------------------------------------------------------------- build


@app.command()
def build(
    seeds: Path = typer.Option(Path("data/seeds.txt"), help="file of seed paper ids"),
    hops: int = typer.Option(2),
    max_papers: Optional[int] = typer.Option(None),
    source: Optional[str] = typer.Option(None, help="s2 | openalex"),
    skip_concepts: bool = typer.Option(False),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(True),
) -> None:
    """Build the corpus and typed graph from seed papers."""
    _setup_logging(verbose)
    # Delegates to the standalone builder so a long build can also be run
    # detached, without the CLI process holding it open.
    from scripts.build_corpus import main as build_main  # type: ignore

    argv = ["--seeds", str(seeds), "--hops", str(hops)]
    if max_papers is not None:
        argv += ["--max-papers", str(max_papers)]
    if source:
        argv += ["--source", source]
    if skip_concepts:
        argv += ["--skip-concepts"]
    if config_path:
        argv += ["--config", config_path]

    saved, sys.argv = sys.argv, ["build_corpus.py"] + argv
    try:
        raise typer.Exit(build_main())
    finally:
        sys.argv = saved


# ------------------------------------------------------------------ search


@app.command()
def search(
    query: str = typer.Argument(..., help="free text, or a paper id in the corpus"),
    top_k: int = typer.Option(20, "--top-k"),
    signals: str = typer.Option("rrf+ppr", help="bm25-only|dense-only|rrf-only|rrf+ppr"),
    output: Optional[Path] = typer.Option(None),
    no_dense: bool = typer.Option(False, "--no-dense"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False),
) -> None:
    """Rank the corpus and print candidates with their provenance."""
    _setup_logging(verbose)
    config, db = _context(config_path)
    retriever = Retriever(db, config, load_dense=not no_dense)

    result = retriever.search(query, top_k=top_k, signals=signals)

    typer.echo(f"corpus: {len(retriever.paper_ids)} papers | signals: {signals}")
    if result.ppr_diagnostics.get("used"):
        typer.echo(
            f"ppr: converged={result.ppr_diagnostics['converged']} "
            f"iterations={result.ppr_diagnostics['iterations']} "
            f"decomposition_error={result.ppr_diagnostics['decomposition_error']:.2e}"
        )
    typer.echo("")
    for candidate in result.candidates:
        typer.echo(
            f"{candidate.rank:>3}. {(candidate.title or '(untitled)')[:64]:<64} "
            f"bm25#{candidate.bm25_rank:<5} dense#{candidate.dense_rank:<5} "
            f"edges={len(candidate.contributing_edges):<3} {candidate.final_score:.4f}"
        )

    path = output or (config.resolve("paths.runs") / f"search_{_slug(query)}.json")
    write_artifact({"search": result.to_dict()}, path, config.to_dict(), db)
    typer.echo(f"\nartifact: {path}")


# ----------------------------------------------------------------- explain


@app.command()
def explain(
    seed: str = typer.Argument(...),
    candidate: str = typer.Argument(...),
    explainer: str = typer.Option("metapath", help="|".join(EXPLAINER_NAMES) + "|all"),
    output: Optional[Path] = typer.Option(None),
    no_dense: bool = typer.Option(True, "--no-dense/--dense"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False),
) -> None:
    """Explain why one candidate was retrieved for a seed."""
    _setup_logging(verbose)
    config, db = _context(config_path)
    retriever = Retriever(db, config, load_dense=not no_dense)
    client = _llm_client(config, db)

    names = _resolve_explainers(explainer)
    explainers = build_explainers(
        retriever.graph, db, client,
        max_paths=int(config.get_path("explain.metapath.max_paths", 12)),
        names=names,
    )
    missing = [n for n in names if n not in explainers]
    if missing:
        typer.secho(
            f"unavailable (needs OPENROUTER_API_KEY): {', '.join(missing)}",
            fg=typer.colors.YELLOW,
        )

    results = {}
    for name, instance in explainers.items():
        explanation = instance.explain(seed, candidate)
        results[name] = explanation.to_dict()
        typer.echo(f"\n=== {name} ===")
        typer.echo(explanation.text)
        typer.echo(f"cited edges: {len(explanation.cited_edges)}")
        for path in explanation.paths[:8]:
            typer.echo(f"  - [{path.relation}] {path.detail}")
        if explanation.unrealised_claims:
            typer.secho(
                f"  unrealised claims: {len(explanation.unrealised_claims)}",
                fg=typer.colors.MAGENTA,
            )

    path = output or (config.resolve("paths.runs") / f"explain_{seed}_{candidate}.json")
    write_artifact(
        {"seed": seed, "candidate": candidate, "explanations": results},
        path, config.to_dict(), db,
    )
    typer.echo(f"\nartifact: {path}")


# ------------------------------------------------------------------ ablate


@app.command()
def ablate(
    seed: str = typer.Argument(...),
    explainer: str = typer.Option("all", help="|".join(EXPLAINER_NAMES) + "|all"),
    top_k: int = typer.Option(20, "--top-k"),
    output: Optional[Path] = typer.Option(None),
    no_dense: bool = typer.Option(False, "--no-dense"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(True),
) -> None:
    """Run the counterfactual edge-ablation protocol. This is the experiment."""
    _setup_logging(verbose)
    config, db = _context(config_path)
    retriever = Retriever(db, config, load_dense=not no_dense)
    client = _llm_client(config, db)

    names = _resolve_explainers(explainer)
    explainers = build_explainers(
        retriever.graph, db, client,
        max_paths=int(config.get_path("explain.metapath.max_paths", 12)),
        names=names,
    )
    if not explainers:
        typer.secho("no explainers available", fg=typer.colors.RED)
        raise typer.Exit(1)

    missing = [n for n in names if n not in explainers]
    if missing:
        typer.secho(
            f"SKIPPING {', '.join(missing)} — OPENROUTER_API_KEY is not set. "
            "Results below cover only the available explainers.",
            fg=typer.colors.YELLOW,
        )

    harness = AblationHarness(retriever, config)
    cells, reports, diagnostics = harness.run(seed, explainers, top_k=top_k)

    typer.echo("")
    _print_ablation_table(reports)

    path = output or (config.resolve("paths.runs") / f"ablate_{seed}.json")
    write_artifact(
        {
            "seed": seed,
            "explainers": list(explainers),
            "skipped_explainers": missing,
            "diagnostics": diagnostics,
            "cells": [c.to_dict() for c in cells],
            "reports": {k: v.to_dict() for k, v in reports.items()},
        },
        path, config.to_dict(), db,
    )
    typer.echo(f"\nartifact: {path}")


def _print_ablation_table(reports) -> None:
    header = (
        f"{'explainer':<14} {'n':>4} {'faithful':>9} "
        f"{'displacement [95% CI]':>28} {'control [95% CI]':>26} {'p':>8}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, report in reports.items():
        typer.echo(
            f"{name:<14} {report.n_testable:>4} "
            f"{report.n_faithful:>4}/{report.n_testable:<4} "
            f"{report.displacement.format(2):>28} "
            f"{report.control_displacement.format(2):>26} "
            f"{report.paired_test.p_value:>8.4f}"
        )


# -------------------------------------------------------------------- tree


@app.command()
def tree(
    seed: str = typer.Argument(...),
    top_k: int = typer.Option(40, "--top-k", help="candidate pool before pruning"),
    ablation: Optional[Path] = typer.Option(
        None, help="ablate artifact supplying necessity scores and verdicts"
    ),
    output: Optional[Path] = typer.Option(None),
    no_dense: bool = typer.Option(False, "--no-dense"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False),
) -> None:
    """Derive the prerequisite DAG and emit the reading-tree artifact."""
    _setup_logging(verbose)
    config, db = _context(config_path)
    retriever = Retriever(db, config, load_dense=not no_dense)

    result = retriever.search(seed, top_k=top_k, signals="rrf+ppr")
    candidates = [c.to_dict() for c in result.candidates]

    necessity: dict[str, float] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    ablation_path = ablation or (config.resolve("paths.runs") / f"ablate_{seed}.json")
    if Path(ablation_path).exists():
        payload = json.loads(Path(ablation_path).read_text(encoding="utf-8"))
        for cell in payload.get("cells", []):
            # The metapath explainer is the deterministic reference, so the tree
            # reports its verdict. LLM verdicts vary per run and would make the
            # layout unstable between sessions.
            if cell.get("explainer") != "metapath" or not cell.get("testable"):
                continue
            necessity[cell["candidate"]] = cell["necessity_score"]
            verdicts[cell["candidate"]] = {
                "ok": bool(cell["faithful"]),
                "from": cell["baseline_rank"],
                "to": cell["ablated_rank"],
                "control": round(float(cell["control_mean"]), 2),
                "note": _verdict_note(cell),
                "necessity": round(float(cell["necessity_score"]), 4),
                "p_value": round(float(cell["p_value"]), 4),
                # The explorer severs exactly these edges when showing an
                # ablation, so what the animation cuts is what the harness cut.
                "cited_edges": cell.get("cited_edges", []),
            }
    else:
        typer.secho(
            f"no ablation artifact at {ablation_path} — tree will have no verdicts "
            "and pruning falls back to retrieval rank.",
            fg=typer.colors.YELLOW,
        )

    why = {
        c.paper_id: [
            (contribution.rel, f"{contribution.src} → {contribution.dst}")
            for contribution in c.edge_contributions[:6]
        ]
        for c in result.candidates
    }

    payload = build_tree(
        db, retriever.graph, seed, candidates,
        necessity=necessity, verdicts=verdicts, why=why,
        max_nodes=int(config.get_path("tree.max_nodes", 15)),
        max_tiers=int(config.get_path("tree.max_tiers", 6)),
        citation_dominance_ratio=float(config.get_path("tree.citation_dominance_ratio", 3.0)),
    )
    payload["provenance"] = provenance(config.to_dict(), db)

    path = Path(output) if output else None
    written = write_tree(payload, path.parent if path else config.resolve("paths.runs"))
    if path and written != path:
        written.replace(path)
        written = path

    typer.echo(
        f"tree: {len(payload['nodes'])} nodes across {len(payload['tiers'])} tiers | "
        f"cycles broken: {len(payload['cycle_drops'])} | "
        f"pruned: {payload['diagnostics']['nodes_pruned']}"
    )
    for drop in payload["cycle_drops"]:
        typer.secho(
            f"  cycle drop: {drop['dropped_edge']['prereq']} -> "
            f"{drop['dropped_edge']['dependent']} ({drop['reason']})",
            fg=typer.colors.YELLOW,
        )
    update_manifest(config.resolve("paths.runs"))
    typer.echo(f"artifact: {written}")


def _verdict_note(cell: dict[str, Any]) -> str:
    """Plain-language verdict. Written for someone who has not read the paper."""
    if cell["faithful"]:
        return "Removing the stated reason pushed this paper down the ranking."
    if cell["displacement"] <= 0:
        return "The stated reason did not affect this ranking."
    return "This paper moved, but no more than removing unrelated links would."


# -------------------------------------------------------------------- eval


@app.command()
def eval(
    dataset: str = typer.Option("primary", help="primary|s2orc|dblp"),
    seeds: Optional[Path] = typer.Option(None, help="seed ids to evaluate over"),
    top_k: int = typer.Option(20, "--top-k"),
    report: bool = typer.Option(True),
    output: Optional[Path] = typer.Option(None),
    no_dense: bool = typer.Option(False, "--no-dense"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(True),
) -> None:
    """One table: NDCG@10 / MRR / necessity / % faithful, by explainer and ablation."""
    _setup_logging(verbose)
    config, db = _context(config_path)
    retriever = Retriever(db, config, load_dense=not no_dense)
    client = _llm_client(config, db)

    seed_ids = _read_seed_ids(seeds, config, db, dataset)
    if not seed_ids:
        typer.secho(f"no seeds available for dataset '{dataset}'", fg=typer.colors.RED)
        raise typer.Exit(1)

    ndcg_k = int(config.get_path("eval.ndcg_k", 10))
    ablations = list(config.get_path("eval.retrieval_ablations") or ["rrf+ppr"])

    # --- retrieval quality, per signal configuration ------------------------
    retrieval_rows = []
    for signals in ablations:
        ndcgs, rrs = [], []
        for seed in seed_ids:
            result = retriever.search(seed, top_k=top_k, signals=signals)
            ranked = [c.paper_id for c in result.candidates]
            relevance = _citation_relevance(db, retriever, seed)
            ndcgs.append(ndcg_at_k(ranked, relevance, ndcg_k))
            rrs.append(reciprocal_rank(ranked, relevance.keys()))
        retrieval_rows.append(
            {
                "signals": signals,
                "ndcg": bootstrap_distribution(ndcgs, seed=int(config.require("seed"))).to_dict(),
                "mrr": bootstrap_distribution(rrs, seed=int(config.require("seed"))).to_dict(),
            }
        )

    # --- faithfulness, per explainer ---------------------------------------
    explainers = build_explainers(
        retriever.graph, db, client,
        max_paths=int(config.get_path("explain.metapath.max_paths", 12)),
    )
    harness = AblationHarness(retriever, config)
    explainer_rows = []
    for name, instance in explainers.items():
        merged = []
        for seed in seed_ids:
            cells, _reports, _diag = harness.run(seed, {name: instance}, top_k=top_k, progress=False)
            merged.extend(cells)
        report_obj = harness.summarise(name, merged)
        explainer_rows.append(report_obj.to_dict())

    if report:
        _print_eval_report(dataset, seed_ids, retrieval_rows, explainer_rows, ndcg_k)

    path = output or (config.resolve("paths.runs") / f"eval_{dataset}.json")
    write_artifact(
        {
            "dataset": dataset,
            "seeds": seed_ids,
            "ndcg_k": ndcg_k,
            "retrieval": retrieval_rows,
            "explainers": explainer_rows,
            "skipped_explainers": [
                n for n in EXPLAINER_NAMES if n not in explainers
            ],
        },
        path, config.to_dict(), db,
    )
    typer.echo(f"\nartifact: {path}")


def _print_eval_report(dataset, seed_ids, retrieval_rows, explainer_rows, ndcg_k) -> None:
    typer.echo(f"\ndataset: {dataset} | seeds: {len(seed_ids)}\n")

    typer.echo("RETRIEVAL")
    header = f"{'signals':<14} {f'NDCG@{ndcg_k} [95% CI]':>30} {'MRR [95% CI]':>30}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in retrieval_rows:
        ndcg, mrr = row["ndcg"], row["mrr"]
        typer.echo(
            f"{row['signals']:<14} "
            f"{ndcg['mean']:.3f} [{ndcg['ci_low']:.3f}, {ndcg['ci_high']:.3f}]".rjust(31)
            + f"  {mrr['mean']:.3f} [{mrr['ci_low']:.3f}, {mrr['ci_high']:.3f}]".rjust(30)
        )

    typer.echo("\nFAITHFULNESS")
    header = (
        f"{'explainer':<14} {'n':>4} {'necessity [95% CI]':>28} "
        f"{'control [95% CI]':>26} {'% faithful':>11} {'p':>8}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in explainer_rows:
        necessity, control = row["necessity"], row["control_necessity"]
        typer.echo(
            f"{row['explainer']:<14} {row['n_testable']:>4} "
            f"{necessity['mean']:.3f} [{necessity['ci_low']:.3f}, {necessity['ci_high']:.3f}]".rjust(29)
            + f"  {control['mean']:.3f} [{control['ci_low']:.3f}, {control['ci_high']:.3f}]".rjust(26)
            + f"  {row['faithful_fraction'] * 100:>9.1f}%"
            + f"  {row['paired_test']['p_value']:>7.4f}"
        )


def _read_seed_ids(seeds: Path | None, config, db: Database, dataset: str) -> list[str]:
    if seeds and Path(seeds).exists():
        ids = []
        for line in Path(seeds).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
        return [i for i in ids if db.has_paper(i)]

    if dataset in ("s2orc", "dblp"):
        rows = db.conn.execute(
            "SELECT paper_id FROM papers WHERE source = ? "
            "ORDER BY citation_count DESC LIMIT 10",
            (dataset,),
        ).fetchall()
        return [r["paper_id"] for r in rows]

    rows = db.conn.execute(
        "SELECT paper_id FROM papers WHERE hop = 0 ORDER BY paper_id"
    ).fetchall()
    return [r["paper_id"] for r in rows]


def _citation_relevance(db: Database, retriever: Retriever, seed: str) -> dict[str, float]:
    """Graded relevance proxy: what the seed cites, and what cites the seed.

    A proxy, not human judgements -- stated plainly because it caps what the
    NDCG column can be claimed to show. It is used only to TUNE retrieval
    (per the guardrails), never to score faithfulness.
    """
    relevance: dict[str, float] = {}
    for rel, neighbour, _edge in retriever.graph.neighbours(seed, ["cites"]):
        relevance[neighbour] = 2.0
    for rel, neighbour, _edge in retriever.graph.neighbours(seed, ["cited_by"]):
        relevance.setdefault(neighbour, 1.0)
    relevance.pop(seed, None)
    return relevance


# ------------------------------------------------------------------- serve


@app.command()
def serve(
    port: int = typer.Option(8000),
    host: str = typer.Option("127.0.0.1"),
    config_path: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """Serve gui/ and runs/ statically. The GUI never recomputes anything."""
    import functools
    import socketserver

    config = load_config(config_path)
    root = Path(config.source).parent

    handler = functools.partial(_ScopedHandler, directory=str(root))
    with socketserver.TCPServer((host, port), handler) as httpd:
        typer.echo(f"serving {'/, '.join(SERVED_PREFIXES)}/ from {root}")
        typer.echo(f"open http://{host}:{port}/gui/")
        typer.echo("ctrl-c to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nstopped")


# The GUI needs exactly these two directories. Serving the project root instead
# would also expose data/synapse.sqlite -- which holds every cached API response
# AND every cached LLM prompt and completion -- plus the source tree, with
# directory listing on. `--host 0.0.0.0` is one flag away from publishing all of
# it, so the allowlist is enforced here rather than left to the default bind.
SERVED_PREFIXES = ("gui", "runs")


class _ScopedHandler(__import__("http.server", fromlist=["SimpleHTTPRequestHandler"]).SimpleHTTPRequestHandler):
    """Static handler restricted to SERVED_PREFIXES."""

    def _permitted(self) -> bool:
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        # Resolve traversal BEFORE matching, so /gui/../data/x.sqlite cannot pass
        # a naive prefix check.
        normalised = posixpath.normpath(unquote(path)).lstrip("/")
        if normalised in ("", "."):
            return True
        head = normalised.split("/", 1)[0]
        return head in SERVED_PREFIXES

    def send_head(self):
        if not self._permitted():
            self.send_error(404, "Not Found")
            return None
        return super().send_head()

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/gui/")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet by default, but errors still surface: a 404 on runs/*.json is
        # exactly the symptom of "why doesn't my run show up in the GUI".
        status = args[1] if len(args) > 1 else ""
        if str(status).startswith(("4", "5")):
            typer.echo(f"  {self.requestline} -> {status}")


# ------------------------------------------------------------------ models


@app.command()
def models(
    free: bool = typer.Option(True, "--free/--all", help="list only zero-cost models"),
    config_path: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """List OpenRouter models currently priced at zero. Works without a key."""
    config, db = _context(config_path)
    client = _llm_client(config, db)
    entries = client.list_free_models()
    if not entries:
        typer.secho("could not reach OpenRouter", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"{len(entries)} free models (configured: {client.model})\n")
    for entry in entries[:30]:
        marker = "*" if entry["id"] == client.model else " "
        typer.echo(f" {marker} {entry['id']:<58} ctx={entry.get('context_length')}")


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    return "".join(keep).strip("-")[:48] or "query"


if __name__ == "__main__":
    app()
