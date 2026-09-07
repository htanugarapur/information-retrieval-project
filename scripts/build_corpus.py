"""Standalone corpus builder.

Exists separately from the CLI because a build against the unauthenticated S2
pool runs for hours and needs to survive being backgrounded, interrupted, and
resumed. Everything it fetches lands in the sqlite cache, so re-running it after
an interruption costs nothing for work already done.

    python scripts/build_corpus.py --seeds data/seeds.txt --hops 2 --max-papers 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synapse.config import load_config, load_credentials  # noqa: E402
from synapse.db import Database  # noqa: E402
from synapse.ingest import build_corpus, build_corpus_openalex, extract_concepts  # noqa: E402
from synapse.openalex import OpenAlexClient  # noqa: E402
from synapse.s2 import SemanticScholarClient  # noqa: E402


def read_seeds(path: Path) -> list[str]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            seeds.append(line)
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Synapse corpus.")
    parser.add_argument("--seeds", default="data/seeds.txt")
    parser.add_argument("--hops", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--skip-concepts", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--source",
        choices=["s2", "openalex"],
        default=None,
        help="override config.ingest_source",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    credentials = load_credentials()
    db = Database(config.resolve("paths.db"))

    seeds_path = Path(args.seeds)
    if not seeds_path.is_absolute():
        seeds_path = Path(config.source).parent / seeds_path
    seeds = read_seeds(seeds_path)
    if not seeds:
        logging.error("no seeds in %s", seeds_path)
        return 1

    source = args.source or config.get_path("ingest_source", "s2")
    hops = args.hops if args.hops is not None else int(config.require("corpus.hops"))
    max_papers = (
        args.max_papers
        if args.max_papers is not None
        else int(config.require("corpus.max_papers"))
    )
    started = time.time()

    if source == "openalex":
        import os

        logging.info("building from %d seeds via OpenAlex (no key required)", len(seeds))
        client = OpenAlexClient(
            db,
            base_url=config.require("openalex.base_url"),
            mailto=os.environ.get("OPENALEX_MAILTO") or config.get_path("openalex.mailto"),
            requests_per_second=float(config.require("openalex.requests_per_second")),
            max_retries=int(config.require("openalex.max_retries")),
            backoff_base_seconds=float(config.require("openalex.backoff_base_seconds")),
            backoff_max_seconds=float(config.require("openalex.backoff_max_seconds")),
            timeout_seconds=int(config.require("openalex.timeout_seconds")),
            seed=int(config.require("seed")),
        )
        stats = build_corpus_openalex(
            db,
            client,
            seeds,
            hops=hops,
            max_papers=max_papers,
            max_refs_per_paper=int(config.require("corpus.max_refs_per_paper")),
            max_cites_per_paper=int(config.require("corpus.max_cites_per_paper")),
        )
    else:
        logging.info(
            "building from %d seeds via Semantic Scholar | api key: %s",
            len(seeds),
            "yes" if credentials.s2_api_key else "NO (shared pool, expect 429s)",
        )
        client = SemanticScholarClient(
            db,
            base_url=config.require("s2.base_url"),
            api_key=credentials.s2_api_key,
            requests_per_second=float(config.require("s2.requests_per_second")),
            max_retries=int(config.require("s2.max_retries")),
            backoff_base_seconds=float(config.require("s2.backoff_base_seconds")),
            backoff_max_seconds=float(config.require("s2.backoff_max_seconds")),
            timeout_seconds=int(config.require("s2.timeout_seconds")),
            seed=int(config.require("seed")),
        )
        stats = build_corpus(
            db,
            client,
            seeds,
            hops=hops,
            max_papers=max_papers,
            max_refs_per_paper=int(config.require("corpus.max_refs_per_paper")),
            max_cites_per_paper=int(config.require("corpus.max_cites_per_paper")),
        )
    logging.info("corpus built in %.1fs: %s", time.time() - started, stats.to_dict())

    if not args.skip_concepts:
        logging.info("extracting concepts ...")
        concept_stats = extract_concepts(
            db,
            top_k=int(config.require("concepts.top_k_per_paper")),
            ngram_max=int(config.require("concepts.ngram_max")),
            dedupe_threshold=float(config.require("concepts.dedupe_threshold")),
            min_papers_per_concept=int(config.require("concepts.min_papers_per_concept")),
        )
        logging.info("concepts: %s", concept_stats)

    summary = {
        "source": source,
        "corpus": stats.to_dict(),
        "elapsed_seconds": round(time.time() - started, 1),
        "papers": db.count_nodes("paper"),
        "authors": db.count_nodes("author"),
        "venues": db.count_nodes("venue"),
        "concepts": db.count_nodes("concept"),
        "edges": db.edge_counts_by_relation(),
    }
    out = config.resolve("paths.runs") / "corpus_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("wrote %s", out)
    print(json.dumps(summary, indent=2))

    db.close()
    if stats.expansion_failed:
        logging.error("BUILD FAILED: corpus has no citation edges (see above)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
