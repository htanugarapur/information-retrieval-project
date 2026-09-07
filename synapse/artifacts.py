"""Run artifacts.

Every command writes a complete, self-describing JSON file to runs/. The GUI
reads these and never recomputes anything, so what a reader sees on screen is
exactly what the experiment produced.

Each artifact embeds the resolved config, the corpus fingerprint, and the code
version. A result that cannot name the corpus and parameters that produced it is
not reproducible, and this is the cheapest possible place to enforce that.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .db import Database


def corpus_fingerprint(db: Database) -> dict[str, Any]:
    """A stable digest of the corpus, so two runs can be compared honestly."""
    counts = {
        "papers": db.count_nodes("paper"),
        "authors": db.count_nodes("author"),
        "venues": db.count_nodes("venue"),
        "concepts": db.count_nodes("concept"),
        "edges": db.edge_counts_by_relation(),
    }
    sources = db.conn.execute(
        "SELECT source, COUNT(*) AS n FROM papers GROUP BY source ORDER BY source"
    ).fetchall()
    counts["sources"] = {row["source"]: int(row["n"]) for row in sources}

    # Digest over sorted edge ids: any change to the graph changes this value.
    digest = hashlib.sha256()
    for row in db.conn.execute("SELECT edge_id FROM edges ORDER BY edge_id"):
        digest.update(row["edge_id"].encode("ascii"))
    counts["edge_digest"] = digest.hexdigest()[:16]
    return counts


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(config: Mapping[str, Any], db: Database) -> dict[str, Any]:
    return {
        "synapse_version": __version__,
        "git_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": dict(config),
        "corpus": corpus_fingerprint(db),
    }


def write_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
    config: Mapping[str, Any] | None = None,
    db: Database | None = None,
) -> Path:
    """Write a run artifact, stamping provenance unless already present."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    body = dict(payload)
    if "provenance" not in body and config is not None and db is not None:
        body["provenance"] = provenance(config, db)

    output.write_text(json.dumps(body, indent=2, default=_fallback), encoding="utf-8")
    return output


def _fallback(value: Any) -> Any:
    """Serialise numpy scalars and dataclass-like objects without silent loss."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into a run artifact")


def read_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_manifest(runs_dir: str | Path) -> Path:
    """Rewrite runs/index.json listing available artifacts.

    A static file server cannot list a directory, so the GUI needs a manifest to
    offer a run picker. Regenerated from what is actually on disk rather than
    appended to, so a deleted run disappears from the GUI instead of 404ing.
    """
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)

    trees, ablations = [], []
    for path in sorted(directory.glob("tree_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        seed_node = next(
            (n for n in payload.get("nodes", []) if n.get("kind") == "seed"), None
        )
        trees.append(
            {
                "file": path.name,
                "seed": payload.get("seed"),
                "title": (seed_node or {}).get("title") or payload.get("seed"),
                "nodes": len(payload.get("nodes", [])),
                "tiers": len(payload.get("tiers", [])),
            }
        )

    for path in sorted(directory.glob("ablate_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ablations.append(
            {
                "file": path.name,
                "seed": payload.get("seed"),
                "explainers": payload.get("explainers", []),
            }
        )

    manifest = directory / "index.json"
    manifest.write_text(
        json.dumps({"trees": trees, "ablations": ablations}, indent=2), encoding="utf-8"
    )
    return manifest
