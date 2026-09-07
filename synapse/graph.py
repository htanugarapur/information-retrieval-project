"""Typed graph: immutable adjacency built once, ablated by producing new copies.

Every stored edge (src, rel, dst) is materialised as TWO directed arcs — forward
and reverse — because relatedness in a citation neighbourhood flows both ways: a
paper is related both to what it cites and to what cites it.

Both arcs carry the id of the single stored edge they came from. That is what
makes ablation honest: deleting one `edge_id` removes both directions, so an
explanation citing "A cites B" cannot keep leaking influence through the reverse
arc after the harness claims to have deleted it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .db import Database

# Reverse-relation naming. Kept explicit rather than derived so that the label a
# human reads in an explanation ("cited_by") is the same string the code routes on.
REVERSE_RELATION = {
    "cites": "cited_by",
    "authored_by": "wrote",
    "published_in": "published",
    "shares_concept": "shares_concept",
}

DEFAULT_EDGE_WEIGHTS = {
    "cites": 1.0,
    "cited_by": 1.0,
    "authored_by": 0.6,
    "wrote": 0.6,
    "published_in": 0.3,
    "published": 0.3,
    "shares_concept": 0.5,
}


def _sealed(array: np.ndarray) -> np.ndarray:
    """Mark an arc buffer read-only.

    `frozen=True` on the dataclass only stops attribute REBINDING; the numpy
    buffers underneath stayed writable, so `graph.arc_weight[i] *= 2` anywhere
    would silently corrupt the baseline graph shared by every subsequent
    ablation in the run -- and the corruption would surface as a plausible
    result, not an error. `retrieve.py` already seals its lexical cache for the
    same reason; this is the structure the causal claim actually rests on.
    """
    if array.base is not None:
        # A view cannot be sealed independently of its parent buffer.
        array = array.copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class TypedGraph:
    """Immutable directed multigraph over papers, authors, venues and concepts.

    Arrays are parallel and arc-indexed. Nothing here is mutated in place;
    `without_edges` / `only_edges` return new graphs, which is what lets the
    ablation harness hold a baseline and a counterfactual side by side without
    any chance of one contaminating the other.
    """

    node_ids: tuple[str, ...]
    node_kinds: tuple[str, ...]
    index: Mapping[str, int]

    arc_src: np.ndarray          # int32[num_arcs]
    arc_dst: np.ndarray          # int32[num_arcs]
    arc_weight: np.ndarray       # float64[num_arcs]
    arc_rel: tuple[str, ...]     # directed relation label per arc
    arc_edge_id: tuple[str, ...] # stored edge each arc derives from

    edge_ids: frozenset[str]

    # ------------------------------------------------------------------ shape

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_arcs(self) -> int:
        return int(self.arc_src.shape[0])

    def has_node(self, node_id: str) -> bool:
        return node_id in self.index

    def node_kind(self, node_id: str) -> str:
        return self.node_kinds[self.index[node_id]]

    def paper_indices(self) -> np.ndarray:
        return np.array(
            [i for i, k in enumerate(self.node_kinds) if k == "paper"], dtype=np.int64
        )

    # ------------------------------------------------------------ construction

    @classmethod
    def from_db(
        cls,
        db: Database,
        edge_weights: Mapping[str, float] | None = None,
        include_edge_ids: Iterable[str] | None = None,
        exclude_edge_ids: Iterable[str] | None = None,
    ) -> "TypedGraph":
        nodes = db.conn.execute(
            "SELECT node_id, kind FROM nodes ORDER BY node_id"
        ).fetchall()
        edges = db.all_edges()
        return cls.from_rows(
            [(r["node_id"], r["kind"]) for r in nodes],
            edges,
            edge_weights=edge_weights,
            include_edge_ids=include_edge_ids,
            exclude_edge_ids=exclude_edge_ids,
        )

    @classmethod
    def from_rows(
        cls,
        nodes: Sequence[tuple[str, str]],
        edges: Sequence[Mapping[str, Any]],
        edge_weights: Mapping[str, float] | None = None,
        include_edge_ids: Iterable[str] | None = None,
        exclude_edge_ids: Iterable[str] | None = None,
    ) -> "TypedGraph":
        weights = dict(DEFAULT_EDGE_WEIGHTS)
        if edge_weights:
            weights.update(edge_weights)

        keep = frozenset(include_edge_ids) if include_edge_ids is not None else None
        drop = frozenset(exclude_edge_ids) if exclude_edge_ids is not None else frozenset()

        node_ids = tuple(n[0] for n in nodes)
        node_kinds = tuple(n[1] for n in nodes)
        index = {node_id: i for i, node_id in enumerate(node_ids)}

        src_list: list[int] = []
        dst_list: list[int] = []
        weight_list: list[float] = []
        rel_list: list[str] = []
        edge_id_list: list[str] = []
        kept_edges: set[str] = set()

        for edge in edges:
            edge_id = edge["edge_id"]
            if edge_id in drop:
                continue
            if keep is not None and edge_id not in keep:
                continue
            src, dst, rel = edge["src"], edge["dst"], edge["rel"]
            if src not in index or dst not in index:
                continue  # dangling reference to a node outside the corpus
            kept_edges.add(edge_id)
            base = float(edge.get("weight", 1.0) or 1.0)
            rev = REVERSE_RELATION.get(rel, rel)

            src_list.append(index[src])
            dst_list.append(index[dst])
            weight_list.append(base * weights.get(rel, 1.0))
            rel_list.append(rel)
            edge_id_list.append(edge_id)

            src_list.append(index[dst])
            dst_list.append(index[src])
            weight_list.append(base * weights.get(rev, weights.get(rel, 1.0)))
            rel_list.append(rev)
            edge_id_list.append(edge_id)

        return cls(
            node_ids=node_ids,
            node_kinds=node_kinds,
            index=index,
            arc_src=_sealed(np.asarray(src_list, dtype=np.int32)),
            arc_dst=_sealed(np.asarray(dst_list, dtype=np.int32)),
            arc_weight=_sealed(np.asarray(weight_list, dtype=np.float64)),
            arc_rel=tuple(rel_list),
            arc_edge_id=tuple(edge_id_list),
            edge_ids=frozenset(kept_edges),
        )

    # -------------------------------------------------------------- ablation

    def without_edges(self, edge_ids: Iterable[str]) -> "TypedGraph":
        """New graph with the given stored edges removed (both arc directions)."""
        drop = frozenset(edge_ids)
        if not drop:
            return self
        mask = np.array([eid not in drop for eid in self.arc_edge_id], dtype=bool)
        return self._masked(mask, self.edge_ids - drop)

    def only_edges(self, edge_ids: Iterable[str]) -> "TypedGraph":
        """New graph keeping ONLY the given stored edges — the sufficiency test."""
        keep = frozenset(edge_ids)
        mask = np.array([eid in keep for eid in self.arc_edge_id], dtype=bool)
        return self._masked(mask, self.edge_ids & keep)

    def _masked(self, mask: np.ndarray, edge_ids: frozenset[str]) -> "TypedGraph":
        if mask.size == 0:
            kept_idx: list[int] = []
        else:
            kept_idx = np.nonzero(mask)[0].tolist()
        return TypedGraph(
            node_ids=self.node_ids,
            node_kinds=self.node_kinds,
            index=self.index,
            arc_src=_sealed(self.arc_src[mask] if mask.size else self.arc_src),
            arc_dst=_sealed(self.arc_dst[mask] if mask.size else self.arc_dst),
            arc_weight=_sealed(self.arc_weight[mask] if mask.size else self.arc_weight),
            arc_rel=tuple(self.arc_rel[i] for i in kept_idx),
            arc_edge_id=tuple(self.arc_edge_id[i] for i in kept_idx),
            edge_ids=edge_ids,
        )

    # ------------------------------------------------------------- neighbours

    def incident_edge_ids(self, node_id: str) -> list[str]:
        """Stored edge ids touching this node — the pool the random control draws from."""
        if node_id not in self.index:
            return []
        i = self.index[node_id]
        touching = (self.arc_src == i) | (self.arc_dst == i)
        seen: dict[str, None] = {}
        for arc_idx in np.nonzero(touching)[0]:
            seen.setdefault(self.arc_edge_id[int(arc_idx)], None)
        return list(seen)

    def neighbours(self, node_id: str, relations: Iterable[str] | None = None) -> list[tuple[str, str, str]]:
        """(relation, neighbour_id, edge_id) triples leaving this node."""
        if node_id not in self.index:
            return []
        rel_filter = frozenset(relations) if relations else None
        i = self.index[node_id]
        result = []
        for arc_idx in np.nonzero(self.arc_src == i)[0]:
            a = int(arc_idx)
            rel = self.arc_rel[a]
            if rel_filter and rel not in rel_filter:
                continue
            result.append((rel, self.node_ids[int(self.arc_dst[a])], self.arc_edge_id[a]))
        return result

    def out_strength(self) -> np.ndarray:
        """Total outgoing arc weight per node. Zero entries are dangling nodes."""
        strength = np.zeros(self.num_nodes, dtype=np.float64)
        if self.num_arcs:
            np.add.at(strength, self.arc_src, self.arc_weight)
        return strength
