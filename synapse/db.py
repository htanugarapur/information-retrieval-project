"""SQLite persistence: API cache, corpus store, typed edge store, LLM cache.

Two jobs, deliberately in one file because they share a connection lifecycle:

1. Cache — an S2 record or LLM completion fetched once is never fetched again.
   This is what makes runs reproducible across days and offline.
2. Corpus — nodes and typed edges, the substrate the whole experiment ablates.

Edges carry a stable string id (`edge_id`). The ablation harness deletes edges
BY ID, and explanations cite edges BY ID, so the two can never drift apart.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 1

# Edge relations. Stored as text rather than an enum table: the set is small,
# fixed, and appears verbatim in run artifacts where a human reads it.
RELATIONS = (
    "cites",
    "authored_by",
    "published_in",
    "shares_concept",
)

NODE_KINDS = ("paper", "author", "venue", "concept")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Raw API responses, keyed by request identity. Never evicted.
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key   TEXT PRIMARY KEY,
    endpoint    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      INTEGER NOT NULL,
    fetched_at  REAL NOT NULL
);

-- LLM completions, keyed by (model, prompt hash). Makes explainers deterministic
-- on replay even though the models themselves are not.
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    completion   TEXT NOT NULL,
    usage        TEXT,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id   TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,
    label     TEXT,
    attrs     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

CREATE TABLE IF NOT EXISTS papers (
    paper_id     TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
    corpus_id    TEXT,
    title        TEXT,
    abstract     TEXT,
    year         INTEGER,
    venue        TEXT,
    citation_count INTEGER DEFAULT 0,
    reference_count INTEGER DEFAULT 0,
    fields       TEXT NOT NULL DEFAULT '[]',
    source       TEXT NOT NULL DEFAULT 's2',
    hop          INTEGER,
    raw          TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);

CREATE TABLE IF NOT EXISTS edges (
    edge_id   TEXT PRIMARY KEY,
    src       TEXT NOT NULL,
    rel       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    attrs     TEXT NOT NULL DEFAULT '{}',
    UNIQUE(src, rel, dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, rel);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, rel);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);
"""


def make_edge_id(src: str, rel: str, dst: str) -> str:
    """Stable, content-derived edge id.

    Derived from the triple rather than a rowid so the same edge gets the same
    id across rebuilds, corpora, and machines. An explanation recorded in one
    run therefore still names real edges in the next.
    """
    digest = hashlib.sha1(f"{src}\x1f{rel}\x1f{dst}".encode("utf-8")).hexdigest()
    return f"e{digest[:16]}"


def hash_prompt(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x1f{prompt}".encode("utf-8")).hexdigest()


class Database:
    """Thin, explicit SQLite wrapper. No ORM — the schema is small and the
    ablation code needs to reason about exact row semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # ---------------------------------------------------------------- lifecycle

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------- api caching

    def get_api_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload, status FROM api_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return {"payload": json.loads(row["payload"]), "status": row["status"]}

    def put_api_cache(
        self, cache_key: str, endpoint: str, payload: Any, status: int
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache"
                "(cache_key, endpoint, payload, status, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (cache_key, endpoint, json.dumps(payload), status, time.time()),
            )

    # ------------------------------------------------------------- llm caching

    def get_llm_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT completion, usage, model FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "completion": row["completion"],
            "usage": json.loads(row["usage"]) if row["usage"] else None,
            "model": row["model"],
        }

    def put_llm_cache(
        self,
        cache_key: str,
        model: str,
        prompt: str,
        completion: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache"
                "(cache_key, model, prompt_hash, prompt, completion, usage, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    model,
                    hash_prompt(model, prompt),
                    prompt,
                    completion,
                    json.dumps(usage) if usage else None,
                    time.time(),
                ),
            )

    # ------------------------------------------------------------------- nodes

    def upsert_node(
        self, node_id: str, kind: str, label: str | None = None, **attrs: Any
    ) -> None:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {kind!r}")
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO nodes(node_id, kind, label, attrs) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(node_id) DO UPDATE SET"
                "   label = COALESCE(excluded.label, nodes.label),"
                "   attrs = excluded.attrs",
                (node_id, kind, label, json.dumps(attrs)),
            )

    def upsert_nodes(self, rows: Iterable[tuple[str, str, str | None, dict[str, Any]]]) -> int:
        payload = [
            (node_id, kind, label, json.dumps(attrs or {}))
            for node_id, kind, label, attrs in rows
        ]
        if not payload:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO nodes(node_id, kind, label, attrs) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(node_id) DO UPDATE SET"
                "   label = COALESCE(excluded.label, nodes.label),"
                "   attrs = excluded.attrs",
                payload,
            )
        return len(payload)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT node_id, kind, label, attrs FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "label": row["label"],
            "attrs": json.loads(row["attrs"]),
        }

    def nodes_by_kind(self, kind: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT node_id, kind, label, attrs FROM nodes WHERE kind = ?", (kind,)
        ).fetchall()
        return [
            {
                "node_id": r["node_id"],
                "kind": r["kind"],
                "label": r["label"],
                "attrs": json.loads(r["attrs"]),
            }
            for r in rows
        ]

    def count_nodes(self, kind: str | None = None) -> int:
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM nodes").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM nodes WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ papers

    def upsert_paper(self, paper: dict[str, Any]) -> None:
        paper_id = paper["paper_id"]
        self.upsert_node(paper_id, "paper", paper.get("title"))
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO papers(paper_id, corpus_id, title, abstract, year, venue,"
                " citation_count, reference_count, fields, source, hop, raw)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(paper_id) DO UPDATE SET"
                "   corpus_id = COALESCE(excluded.corpus_id, papers.corpus_id),"
                "   title = COALESCE(excluded.title, papers.title),"
                "   abstract = COALESCE(excluded.abstract, papers.abstract),"
                "   year = COALESCE(excluded.year, papers.year),"
                "   venue = COALESCE(excluded.venue, papers.venue),"
                "   citation_count = MAX(excluded.citation_count, papers.citation_count),"
                "   reference_count = MAX(excluded.reference_count, papers.reference_count),"
                "   fields = excluded.fields,"
                "   hop = MIN(COALESCE(excluded.hop, 99), COALESCE(papers.hop, 99))",
                (
                    paper_id,
                    paper.get("corpus_id"),
                    paper.get("title"),
                    paper.get("abstract"),
                    paper.get("year"),
                    paper.get("venue"),
                    int(paper.get("citation_count") or 0),
                    int(paper.get("reference_count") or 0),
                    json.dumps(paper.get("fields") or []),
                    paper.get("source", "s2"),
                    paper.get("hop"),
                    json.dumps(paper.get("raw")) if paper.get("raw") is not None else None,
                ),
            )

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return _paper_row(row) if row else None

    def all_papers(self, source: str | None = None) -> list[dict[str, Any]]:
        if source:
            rows = self._conn.execute(
                "SELECT * FROM papers WHERE source = ? ORDER BY paper_id", (source,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM papers ORDER BY paper_id").fetchall()
        return [_paper_row(r) for r in rows]

    def has_paper(self, paper_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1", (paper_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------- edges

    def add_edge(
        self, src: str, rel: str, dst: str, weight: float = 1.0, **attrs: Any
    ) -> str:
        if rel not in RELATIONS:
            raise ValueError(f"unknown relation: {rel!r}")
        edge_id = make_edge_id(src, rel, dst)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO edges(edge_id, src, rel, dst, weight, attrs)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(edge_id) DO UPDATE SET"
                "   weight = excluded.weight, attrs = excluded.attrs",
                (edge_id, src, rel, dst, weight, json.dumps(attrs)),
            )
        return edge_id

    def add_edges(self, triples: Sequence[tuple[str, str, str, float, dict[str, Any]]]) -> list[str]:
        payload = []
        edge_ids = []
        for src, rel, dst, weight, attrs in triples:
            if rel not in RELATIONS:
                raise ValueError(f"unknown relation: {rel!r}")
            edge_id = make_edge_id(src, rel, dst)
            edge_ids.append(edge_id)
            payload.append((edge_id, src, rel, dst, weight, json.dumps(attrs or {})))
        if not payload:
            return []
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO edges(edge_id, src, rel, dst, weight, attrs)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(edge_id) DO UPDATE SET"
                "   weight = excluded.weight, attrs = excluded.attrs",
                payload,
            )
        return edge_ids

    def all_edges(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT edge_id, src, rel, dst, weight, attrs FROM edges ORDER BY edge_id"
        ).fetchall()
        return [_edge_row(r) for r in rows]

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT edge_id, src, rel, dst, weight, attrs FROM edges WHERE edge_id = ?",
            (edge_id,),
        ).fetchone()
        return _edge_row(row) if row else None

    def count_edges(self, rel: str | None = None) -> int:
        if rel is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE rel = ?", (rel,)
            ).fetchone()
        return int(row["n"])

    def edge_counts_by_relation(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT rel, COUNT(*) AS n FROM edges GROUP BY rel ORDER BY rel"
        ).fetchall()
        return {r["rel"]: int(r["n"]) for r in rows}


def _paper_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "corpus_id": row["corpus_id"],
        "title": row["title"] or "",
        "abstract": row["abstract"] or "",
        "year": row["year"],
        "venue": row["venue"] or "",
        "citation_count": row["citation_count"] or 0,
        "reference_count": row["reference_count"] or 0,
        "fields": json.loads(row["fields"]) if row["fields"] else [],
        "source": row["source"],
        "hop": row["hop"],
    }


def _edge_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "edge_id": row["edge_id"],
        "src": row["src"],
        "rel": row["rel"],
        "dst": row["dst"],
        "weight": row["weight"],
        "attrs": json.loads(row["attrs"]) if row["attrs"] else {},
    }
