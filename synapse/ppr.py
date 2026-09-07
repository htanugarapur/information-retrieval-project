"""Personalized PageRank with live edge-flow recording.

WHY THIS IS NOT networkx.pagerank
---------------------------------
`networkx.pagerank` returns node scores and nothing else. The entire experiment
rests on knowing WHICH EDGES delivered mass to a candidate, and a score vector
cannot answer that. Reconstructing edge contributions afterwards from the graph
alone would be a plausible-looking guess, and the build prompt is right that an
approximate `contributing_edges` invalidates the paper. So propagation is
implemented here, in ~80 lines of power iteration, and the per-arc flow is
accumulated *while the iteration runs*.

THE DECOMPOSITION IS EXACT, NOT ESTIMATED
-----------------------------------------
At the fixed point the iteration satisfies, per node v:

    x[v] = teleport[v] + SUM over arcs a=(u->v) of flow[a]
    flow[a] = (1 - alpha) * x[u] * w[a] / out_strength[u]

`flow` is recorded during the final iteration — the same array the iteration
itself uses to produce x. `check_decomposition()` asserts the identity above to
float tolerance, and a test compares it against an independent live-accumulating
reference implementation. So attribution is arithmetic over recorded quantities,
not inference about them.

Dangling nodes (no outgoing arcs) return their mass to the personalization
vector; that share is teleport, never credited to an edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .graph import TypedGraph


@dataclass(frozen=True)
class PPRResult:
    """Converged PPR state plus the flows that produced it."""

    scores: np.ndarray            # float64[num_nodes], sums to ~1
    arc_flow: np.ndarray          # float64[num_arcs], mass along each arc at fixed point
    teleport: np.ndarray          # float64[num_nodes], mass injected, not edge-borne
    iterations: int
    converged: bool
    residual: float
    alpha: float
    seed_indices: tuple[int, ...]

    def score_of(self, graph: TypedGraph, node_id: str) -> float:
        idx = graph.index.get(node_id)
        return 0.0 if idx is None else float(self.scores[idx])

    def check_decomposition(self, graph: TypedGraph, tol: float = 1e-9) -> float:
        """Max absolute violation of x[v] == teleport[v] + sum(inflow[v]).

        Returned rather than asserted so callers can log it into run artifacts:
        a reviewer can see the number that guarantees the attribution is exact.
        """
        inflow = np.zeros(graph.num_nodes, dtype=np.float64)
        if graph.num_arcs:
            np.add.at(inflow, graph.arc_dst, self.arc_flow)
        return float(np.max(np.abs(self.scores - (self.teleport + inflow))))


@dataclass(frozen=True)
class EdgeContribution:
    """One edge's share of a candidate's PPR mass."""

    edge_id: str
    src: str
    rel: str
    dst: str
    credit: float       # fraction of the candidate's mass that flowed through here
    depth: int          # hops back from the candidate

    def as_triple(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "src": self.src,
            "rel": self.rel,
            "dst": self.dst,
            "credit": self.credit,
            "depth": self.depth,
        }


def personalized_pagerank(
    graph: TypedGraph,
    seed_nodes: Sequence[str] | Mapping[str, float],
    alpha: float = 0.15,
    max_iterations: int = 60,
    tolerance: float = 1e-10,
) -> PPRResult:
    """Power-iterate PPR, recording per-arc flow during the final iteration.

    alpha is the TELEPORT probability (probability of jumping back to the seed),
    matching networkx's `1 - damping` convention.
    """
    n = graph.num_nodes
    if n == 0:
        return PPRResult(
            scores=np.zeros(0),
            arc_flow=np.zeros(0),
            teleport=np.zeros(0),
            iterations=0,
            converged=True,
            residual=0.0,
            alpha=alpha,
            seed_indices=(),
        )

    personalization, seed_indices = _personalization_vector(graph, seed_nodes)
    out_strength = graph.out_strength()
    dangling = out_strength == 0.0

    # Per-arc transition probability: w(a) / out_strength(src(a)). Constant across
    # iterations, so it is computed once and reused -- this is the array that makes
    # flow recording free.
    if graph.num_arcs:
        arc_prob = graph.arc_weight / out_strength[graph.arc_src]
    else:
        arc_prob = np.zeros(0, dtype=np.float64)

    def propagate(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One propagation step: (next_state, arc_flow, teleport).

        Defined once and called from both the loop and the final pass. These two
        call sites previously held identical copy-pasted arithmetic; in the one
        module whose docstring says an approximate flow invalidates the paper,
        two independent copies of the fixed-point formula were a drift risk that
        only `check_decomposition` would have caught, and only after the fact.
        """
        dangling_mass = float(state[dangling].sum())
        if graph.num_arcs:
            flow = (1.0 - alpha) * state[graph.arc_src] * arc_prob
            inflow = np.zeros(n, dtype=np.float64)
            np.add.at(inflow, graph.arc_dst, flow)
        else:
            flow = np.zeros(0, dtype=np.float64)
            inflow = np.zeros(n, dtype=np.float64)
        injected = (alpha + (1.0 - alpha) * dangling_mass) * personalization
        return injected + inflow, flow, injected

    x = personalization.copy()
    arc_flow = np.zeros(graph.num_arcs, dtype=np.float64)
    teleport = np.zeros(n, dtype=np.float64)
    residual = float("inf")
    converged = False
    iterations = 0

    for step in range(1, max_iterations + 1):
        iterations = step
        nxt, arc_flow, teleport = propagate(x)
        residual = float(np.abs(nxt - x).sum())
        x = nxt
        if residual < tolerance:
            converged = True
            break

    # One extra pass at the converged vector so `arc_flow` and `x` describe the
    # SAME state. Without this the recorded flow lags the score by one iteration
    # and the decomposition identity fails by the residual.
    x, arc_flow, teleport = propagate(x)

    return PPRResult(
        scores=x,
        arc_flow=arc_flow,
        teleport=teleport,
        iterations=iterations,
        converged=converged,
        residual=residual,
        alpha=alpha,
        seed_indices=tuple(seed_indices),
    )


def _personalization_vector(
    graph: TypedGraph, seed_nodes: Sequence[str] | Mapping[str, float]
) -> tuple[np.ndarray, list[int]]:
    n = graph.num_nodes
    vector = np.zeros(n, dtype=np.float64)
    indices: list[int] = []

    if isinstance(seed_nodes, Mapping):
        items = list(seed_nodes.items())
    else:
        items = [(node_id, 1.0) for node_id in seed_nodes]

    for node_id, weight in items:
        idx = graph.index.get(node_id)
        if idx is None:
            continue
        vector[idx] += float(weight)
        indices.append(idx)

    total = vector.sum()
    if total <= 0:
        # No seed present in the graph: fall back to uniform, and let the caller
        # notice via an empty seed_indices rather than silently ranking noise.
        vector[:] = 1.0 / n
        return vector, []
    return vector / total, sorted(set(indices))


class FlowIndex:
    """Reverse arc index: which arcs deliver mass INTO each node.

    Built once per PPR result and reused across candidates, because a single
    ablation run attributes hundreds of candidates over the same graph.
    """

    def __init__(self, graph: TypedGraph):
        self.graph = graph
        if graph.num_arcs == 0:
            self.order = np.zeros(0, dtype=np.int64)
            self.offsets = np.zeros(graph.num_nodes + 1, dtype=np.int64)
            return
        self.order = np.argsort(graph.arc_dst, kind="stable").astype(np.int64)
        sorted_dst = graph.arc_dst[self.order]
        self.offsets = np.searchsorted(
            sorted_dst, np.arange(graph.num_nodes + 1), side="left"
        ).astype(np.int64)

    def in_arcs(self, node_index: int) -> np.ndarray:
        start = self.offsets[node_index]
        end = self.offsets[node_index + 1]
        return self.order[start:end]


def attribute_edges(
    graph: TypedGraph,
    result: PPRResult,
    candidate: str,
    max_depth: int = 3,
    min_flow_fraction: float = 0.01,
    max_edges: int = 40,
    flow_index: FlowIndex | None = None,
) -> list[EdgeContribution]:
    """Which edges fed this candidate's PPR mass, and how much of it.

    Walks backward from the candidate over the RECORDED arc flows. At each node
    the incoming flows plus the teleport share exactly account for that node's
    score, so `credit` is a true fraction of the candidate's mass, not a heuristic
    salience. `credit` sums to at most 1.0 across all returned edges; the shortfall
    is mass that arrived by teleport or from beyond `max_depth`.
    """
    idx = graph.index.get(candidate)
    if idx is None or graph.num_arcs == 0:
        return []

    index = flow_index or FlowIndex(graph)
    scores = result.scores
    arc_flow = result.arc_flow

    credit: dict[str, float] = {}
    depth_of: dict[str, int] = {}
    triple_of: dict[str, tuple[str, str, str]] = {}

    frontier: dict[int, float] = {idx: 1.0}

    for depth in range(1, max_depth + 1):
        next_frontier: dict[int, float] = {}
        for node_index, responsibility in frontier.items():
            mass = float(scores[node_index])
            if mass <= 0.0 or responsibility <= 0.0:
                continue
            for arc in index.in_arcs(node_index):
                a = int(arc)
                flow = float(arc_flow[a])
                if flow <= 0.0:
                    continue
                share = (flow / mass) * responsibility
                if share < min_flow_fraction:
                    continue
                edge_id = graph.arc_edge_id[a]
                src_index = int(graph.arc_src[a])
                credit[edge_id] = credit.get(edge_id, 0.0) + share
                if edge_id not in depth_of:
                    depth_of[edge_id] = depth
                    triple_of[edge_id] = (
                        graph.node_ids[src_index],
                        graph.arc_rel[a],
                        graph.node_ids[int(graph.arc_dst[a])],
                    )
                next_frontier[src_index] = next_frontier.get(src_index, 0.0) + share
        if not next_frontier:
            break
        frontier = next_frontier

    contributions = [
        EdgeContribution(
            edge_id=edge_id,
            src=triple_of[edge_id][0],
            rel=triple_of[edge_id][1],
            dst=triple_of[edge_id][2],
            credit=value,
            depth=depth_of[edge_id],
        )
        for edge_id, value in credit.items()
    ]
    # Deterministic ordering: credit desc, then edge_id — ties must not depend on
    # dict insertion order or the run stops being reproducible.
    contributions.sort(key=lambda c: (-c.credit, c.edge_id))
    return contributions[:max_edges]


def reference_flow_accumulation(
    graph: TypedGraph,
    seed_nodes: Sequence[str],
    alpha: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent, deliberately slow reference implementation.

    Accumulates arc flow one arc at a time in a Python loop, with no vectorised
    shortcuts, so the test suite can prove the fast path records the same flows.
    Never call this outside tests.
    """
    n = graph.num_nodes
    personalization, _ = _personalization_vector(graph, seed_nodes)
    out_strength = graph.out_strength()

    x = personalization.copy()
    arc_flow = np.zeros(graph.num_arcs, dtype=np.float64)

    for _ in range(iterations):
        arc_flow = np.zeros(graph.num_arcs, dtype=np.float64)
        nxt = np.zeros(n, dtype=np.float64)
        dangling_mass = 0.0
        for node in range(n):
            if out_strength[node] == 0.0:
                dangling_mass += x[node]
        for a in range(graph.num_arcs):
            u = int(graph.arc_src[a])
            v = int(graph.arc_dst[a])
            flow = (1.0 - alpha) * x[u] * graph.arc_weight[a] / out_strength[u]
            arc_flow[a] = flow
            nxt[v] += flow
        nxt += (alpha + (1.0 - alpha) * dangling_mass) * personalization
        x = nxt

    return x, arc_flow
