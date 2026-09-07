# Synapse

A scholarly information retrieval system built as a **measurement instrument**,
not a SOTA ranker.

The research contribution is not ranking accuracy. It is a **counterfactual
edge-ablation protocol** that tests whether a retrieval system's stated
explanation actually caused the ranking it claims to explain.

Systems that explain their recommendations — "shares an author", "cited by the
same work" — present those statements as accounts of the ranking, and readers
treat them as causal. But the explainer is usually a separate module from the
ranker, and nothing in a standard evaluation checks that the two agree. An
explanation can be *true* and still be a *non-explanation*.

## The protocol

For each (seed, candidate, explainer):

1. Record the baseline rank **r₀**.
2. Delete **exactly** the edges the explanation cited.
3. Recompute PPR and re-rank with identical parameters → **r₁**.
4. Compare the displacement against a **count-matched random-edge control**.

An explanation is *faithful* only where its displacement beats its own control
distribution at p < 0.05. Everything else is reported as unfaithful — including
explanations that cite nothing.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

.venv/bin/synapse build  --seeds data/seeds.txt --hops 2   # corpus + typed graph
.venv/bin/synapse ablate W3015883388 --explainer all       # the experiment
.venv/bin/synapse tree   W3015883388                       # reading-tree artifact
.venv/bin/synapse serve                                    # GUI at /gui/
```

`build` uses OpenAlex by default and needs no API key. See
[docs/CORPORA.md](docs/CORPORA.md) for why, and for the Semantic Scholar path.

The two LLM explainers need an OpenRouter key:

```bash
export OPENROUTER_API_KEY=...      # https://openrouter.ai/keys
.venv/bin/synapse models --free    # list currently-free models
```

Without it, `llm-grounded` and `llm-free` are **skipped with a warning** and
recorded as skipped in the artifact. They are never silently replaced by the
deterministic explainer.

## Commands

| Command | Purpose |
|---|---|
| `synapse build` | Expand a citation neighbourhood into a typed graph |
| `synapse search <query\|paper-id>` | Rank with full per-candidate provenance |
| `synapse explain <seed> <candidate>` | Run one or all explainers |
| `synapse ablate <seed>` | **The experiment.** Ablation + control + sufficiency |
| `synapse tree <seed>` | Derive the prerequisite DAG for the GUI |
| `synapse eval --dataset <name>` | NDCG / MRR / necessity / % faithful |
| `synapse serve` | Static server for `gui/` over `runs/` |
| `synapse models` | List free OpenRouter models |

## Layout

```
synapse/
  ppr.py         Personalized PageRank with LIVE per-arc flow recording
  graph.py       Immutable typed graph; ablation returns new graphs
  retrieve.py    BM25 + dense + RRF + PPR, per-signal attribution
  explain.py     Three explainers, differing only in what they may see
  ablate.py      The contribution: control first, then necessity/sufficiency
  metrics.py     Bootstrap CIs and paired permutation tests
  tree.py        Prerequisite DAG derivation (Phase 6a)
  ingest.py      Corpus construction; S2ORC and DBLP loaders
  openalex.py    OpenAlex client   ─┐ same schema, same typed edges
  s2.py          Semantic Scholar  ─┘
gui/
  synapse-tree.html      PRIMARY view — tiered reading tree
  synapse-explorer.html  SECONDARY view — neuron graph
paper/
  paper.md               Scaffold; §5–6 wait for the numbers
  generate_results.py    Builds results.md FROM runs/ — never hand-typed
```

## Why PPR is implemented here rather than imported

`networkx.pagerank` returns node scores and nothing else. The entire experiment
rests on knowing **which edges delivered mass to a candidate**, and a score
vector cannot answer that. Reconstructing edge contributions afterwards would be
a plausible-looking guess.

So propagation is ~80 lines of power iteration with per-arc flow recorded *while
the iteration runs*. At the fixed point the recorded flows satisfy, exactly:

```
x(v) = teleport(v) + Σ over in-arcs a of flow(a)
```

The residual of this identity is written into every run artifact
(**observed: 0.0**) and is asserted in the test suite against an independent
naive-loop reference implementation at 1e-12 tolerance. Attribution is arithmetic
over recorded quantities, not inference about them.

## Guardrails enforced in code

- **The random control runs on every cell**, never optionally. Cited edges stay
  in the control pool: the null under test is exchangeability, and excluding them
  would test a weaker null and bias the protocol toward whichever explainer
  cites least.
- **`llm-free` never sees the graph.** Asserted mechanically — a test scans the
  prompt for every edge id, node id, author name, venue name, and meta-path
  phrase in the corpus.
- **Empty explanations are `untestable`, not displacement-zero.** Scoring them as
  zero would launder an empty explanation into a harmless one.
- **The tree's prerequisite edges come from `tree.py`, never an LLM.** An
  LLM-authored tree is unfalsifiable and cannot be ablated.
- **No mean without spread.** `Distribution` carries its bootstrap CI, so no code
  path can print a centre without one.
- **Trial-count sanity check.** With *n* control trials the smallest achievable
  p-value is 1/(n+1); the harness refuses to start if that floor is not below α.

## Determinism

Fixed seed. All API and LLM responses cached to sqlite, including 404s and
backoff jitter. Every artifact embeds the resolved config, a corpus edge digest,
and the code revision. A replay is deterministic and offline.

## Tests

```bash
.venv/bin/python -m pytest -q --cov=synapse
```

215 tests, 85% coverage. The strictest are in `tests/test_ppr.py` (flow
exactness) and `tests/test_ablate.py`, whose decisive case checks the harness can
separate a candidate whose rank is genuinely caused by citation structure from
one that ranks on text while sharing only a crowded venue.

`tests/test_gui_smoke.py` drives a real Chromium and asserts the WCAG floor:
arrow-key traversal, visible focus rings, `prefers-reduced-motion`, and that no
node state is conveyed by colour alone. It also asserts that an **unfaithful
verdict renders at exactly the same size as a faithful one**.

## Status

Phases 1–7 built and running on a real 500-paper OpenAlex corpus.

Measured (metapath explainer, 5 seeds, 100 candidates):
displacement **14.6 [11.2, 18.1]** vs random control **2.4 [1.5, 3.4]**,
necessity **0.028 [0.023, 0.034]** vs control **0.003 [0.002, 0.004]**,
**58% of candidates faithful**, paired permutation p = 0.0001.

The LLM explainer columns are empty pending an `OPENROUTER_API_KEY`, and
`paper/results.md` says so at the top rather than presenting a one-explainer
table as a comparison.
