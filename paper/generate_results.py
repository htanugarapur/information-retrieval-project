"""Generate the paper's results tables from runs/.

The tables are BUILT FROM ARTIFACTS, never typed by hand. If a number appears in
the paper it can be traced to a run file, and regenerating after a new run
updates the paper rather than leaving a stale claim behind.

    python paper/generate_results.py            # writes paper/results.md
    python paper/generate_results.py --check    # non-zero if regeneration would change it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUTPUT = ROOT / "paper" / "results.md"


# Artifacts that exist on disk but could not be read. A truncated ablate_*.json
# (interrupted run, disk full) previously vanished from every table with no
# footnote, so "ran and was silently discarded" looked identical to "never run".
UNREADABLE: list[str] = []


def load(pattern: str) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(RUNS.glob(pattern)):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read {path.name}: {exc}", file=sys.stderr)
            UNREADABLE.append(path.name)
    return payloads


def fmt(distribution: Mapping[str, Any] | None, digits: int = 2) -> str:
    """Every centre carries its interval. There is no code path that prints a
    bare mean, because a mean without spread is what this paper argues against."""
    if not distribution or not distribution.get("n"):
        return "—"
    return (
        f"{distribution['mean']:.{digits}f} "
        f"[{distribution['ci_low']:.{digits}f}, {distribution['ci_high']:.{digits}f}]"
    )


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_No runs available._\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def faithfulness_table(ablations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for payload in ablations:
        seed = payload.get("seed", "?")
        for name, report in (payload.get("reports") or {}).items():
            rows.append([
                seed,
                name,
                report["n_testable"],
                fmt(report["displacement"]),
                fmt(report["control_displacement"]),
                fmt(report["necessity"], 3),
                f"{report['faithful_fraction'] * 100:.0f}%",
                f"{report['paired_test']['p_value']:.4f}",
            ])
    return table(
        ["seed", "explainer", "n", "displacement [95% CI]",
         "random control [95% CI]", "necessity [95% CI]", "% faithful", "p"],
        rows,
    )


def retrieval_table(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for payload in evaluations:
        dataset = payload.get("dataset", "?")
        for row in payload.get("retrieval") or []:
            rows.append([
                dataset, row["signals"],
                fmt(row["ndcg"], 3), fmt(row["mrr"], 3),
            ])
    return table(["corpus", "signals", "NDCG@10 [95% CI]", "MRR [95% CI]"], rows)


def eval_faithfulness_table(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for payload in evaluations:
        dataset = payload.get("dataset", "?")
        for row in payload.get("explainers") or []:
            rows.append([
                dataset, row["explainer"], row["n_testable"],
                fmt(row["necessity"], 3), fmt(row["control_necessity"], 3),
                f"{row['faithful_fraction'] * 100:.0f}%",
                f"{row['paired_test']['p_value']:.4f}",
                row["empty_explanations"],
                row["unrealised_claim_count"],
            ])
    return table(
        ["corpus", "explainer", "n", "necessity [95% CI]", "control [95% CI]",
         "% faithful", "p", "empty expl.", "unrealised claims"],
        rows,
    )


def corpus_table(payloads: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    seen = set()
    for payload in payloads:
        corpus = (payload.get("provenance") or {}).get("corpus")
        if not corpus:
            continue
        key = corpus.get("edge_digest")
        if key in seen:
            continue
        seen.add(key)
        edges = corpus.get("edges", {})
        rows.append([
            ", ".join(f"{k}:{v}" for k, v in (corpus.get("sources") or {}).items()) or "—",
            corpus.get("papers", 0), corpus.get("authors", 0),
            corpus.get("venues", 0), corpus.get("concepts", 0),
            edges.get("cites", 0), edges.get("shares_concept", 0),
            key or "—",
        ])
    return table(
        ["source", "papers", "authors", "venues", "concepts",
         "cites", "shares_concept", "edge digest"],
        rows,
    )


def limitations(trees: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    total_drops = 0
    related_work_missing = 0
    for tree in trees:
        drops = tree.get("cycle_drops") or []
        total_drops += len(drops)
        for drop in drops:
            edge = drop["dropped_edge"]
            lines.append(
                f"- `{edge['prereq']}` → `{edge['dependent']}` "
                f"(rule: {drop['reason']}, weight {drop['weight']})"
            )
        if not (tree.get("diagnostics") or {}).get("related_work_available"):
            related_work_missing += 1

    body = [f"Cycle-breaking edge drops across {len(trees)} tree(s): **{total_drops}**.\n"]
    if lines:
        body.append("Edges removed to make the prerequisite graph acyclic:\n")
        body.extend(lines)
        body.append("")
    else:
        body.append(
            "No cycles were encountered in the current trees. This is a property of "
            "these particular neighbourhoods, not a guarantee: citation graphs "
            "contain cycles (preprint/published pairs, simultaneous submissions) "
            "and the harness logs every drop when they occur.\n"
        )
    if related_work_missing:
        body.append(
            f"Prerequisite rule 3 (related-work membership) contributed **no edges** in "
            f"{related_work_missing} of {len(trees)} tree(s): neither OpenAlex nor the "
            "Semantic Scholar Graph API serves section-segmented full text, so the rule "
            "is implemented but inert. Trees here rest on rules 1 and 2 only.\n"
        )
    return "\n".join(body)


def build() -> str:
    ablations = load("ablate_*.json")
    evaluations = load("eval_*.json")
    trees = load("tree_*.json")
    everything = ablations + evaluations

    skipped = sorted({
        name
        for payload in everything
        for name in (payload.get("skipped_explainers") or [])
    })

    parts = [
        "# Results",
        "",
        "<!-- GENERATED FILE — do not edit by hand.",
        "     Regenerate with: python paper/generate_results.py",
        "     Every number below is read from a run artifact in runs/. -->",
        "",
        f"Artifacts: {len(ablations)} ablation run(s), {len(evaluations)} evaluation run(s), "
        f"{len(trees)} tree(s).",
        "",
    ]

    if skipped:
        parts += [
            f"> **Incomplete coverage.** These explainers were not run and are absent "
            f"from every table below: `{'`, `'.join(skipped)}`. "
            "They require `OPENROUTER_API_KEY`. No result here should be read as a "
            "comparison across all three explainers.",
            "",
        ]

    if UNREADABLE:
        parts += [
            f"> **Artifacts excluded.** {len(UNREADABLE)} run file(s) on disk could not "
            f"be parsed and contribute to no table below: "
            f"`{'`, `'.join(sorted(set(UNREADABLE)))}`. "
            "Re-run them before citing these results; a run that was discarded is "
            "not the same as a run that never happened.",
            "",
        ]

    parts += [
        "## Table 1 — Corpora",
        "",
        corpus_table(everything),
        "",
        "## Table 2 — Retrieval quality",
        "",
        "> **This table is partly circular and must not be read as evidence that the "
        "graph signal is better.** Relevance here is a citation-based proxy (papers the "
        "seed cites, and papers citing the seed) because no human judgements exist for "
        "this corpus. Personalized PageRank propagates along those same citation edges, "
        "so the `rrf+ppr` row is scored against a target its own signal is built from. "
        "The gap between `rrf+ppr` and `rrf-only` is therefore inflated by construction "
        "and is reported only to show the retrieval configurations are distinguishable, "
        "not to claim ranking quality. Nothing in the faithfulness analysis depends on "
        "this table.",
        "",
        retrieval_table(evaluations),
        "",
        "## Table 3 — Explanation faithfulness, per seed",
        "",
        "Displacement is the change in rank after deleting exactly the edges the "
        "explanation cited. The random control deletes the same NUMBER of edges drawn "
        "from the same incident pool. An explanation is faithful only where its "
        "displacement beats its own control distribution at p < 0.05.",
        "",
        faithfulness_table(ablations),
        "",
        "## Table 4 — Explanation faithfulness, aggregated",
        "",
        eval_faithfulness_table(evaluations),
        "",
        "## Limitations recorded by the harness",
        "",
        limitations(trees),
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the file is out of date")
    args = parser.parse_args()

    content = build()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            print("results.md is out of date; run: python paper/generate_results.py")
            return 1
        print("results.md is up to date")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
