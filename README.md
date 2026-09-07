<div align="center">

# ⚡ Synapse

### Every retrieval system tells you *why*. Almost none can prove it.

**A measurement instrument for explanation faithfulness — not another ranker.**

`Python 3.11+` · `215 tests` · `85% coverage` · `deterministic & offline-replayable`

</div>

---

## The problem, in one example

A search engine recommends a paper and tells you:

> *"Recommended because it shares an author with your seed."*

The shared author is real. You can verify it. And yet — the ranking may have come
entirely from text similarity, with that author edge contributing **nothing**.

The explanation is **true**. It is also a **non-explanation**.

This happens because the explainer is almost always a *different module* from the
ranker, and nothing in a standard evaluation ever checks that the two agree.
NDCG cannot catch it. Neither can a human reading the sentence.

**Synapse catches it.**

---

## The protocol

For every `(seed, candidate, explainer)` triple:

```
  ①  Record the baseline rank                                        r₀
  ②  Delete EXACTLY the edges the explanation cited
  ③  Recompute PPR, re-rank, identical parameters                    r₁
  ④  Compare |r₁ − r₀| against a count-matched RANDOM-edge control
```

> An explanation is **faithful** only where its displacement beats **its own**
> control distribution at *p* < 0.05.
> Everything else is reported as unfaithful — **including explanations that cite nothing.**

That last clause is the whole discipline. An empty explanation is recorded as
`untestable`, never as displacement-zero, because scoring it zero would launder
"I said nothing" into "I said something harmless".

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

.venv/bin/synapse build  --seeds data/seeds.txt --hops 2   # corpus + typed graph
.venv/bin/synapse ablate W3015883388 --explainer all       # ← the experiment
.venv/bin/synapse tree   W3015883388                       # reading-tree artifact
.venv/bin/synapse serve                                    # GUI at /gui/
```

`build` uses **OpenAlex** by default and needs **no API key**.
See [docs/CORPORA.md](docs/CORPORA.md) for why, and for the Semantic Scholar path.

<details>
<summary><b>Enabling the two LLM explainers</b> (optional — needs an OpenRouter key)</summary>

```bash
export OPENROUTER_API_KEY=...      # https://openrouter.ai/keys
.venv/bin/synapse models --free    # list currently-free models
```

Without the key, `llm-grounded` and `llm-free` are **skipped with a warning** and
recorded as skipped in the artifact. They are *never* silently replaced by the
deterministic explainer.
</details>

---

## Where it stands today

Phases 1–7 built and running on a real **500-paper OpenAlex corpus**.

| metric — `metapath`, 5 seeds, 100 candidates | measured | random control |
|---|---|---|
| **Displacement** | **14.6** `[11.2, 18.1]` | 2.4 `[1.5, 3.4]` |
| **Necessity** | **0.028** `[0.023, 0.034]` | 0.003 `[0.002, 0.004]` |
| **Candidates faithful** | **58%** | — |
| Paired permutation | *p* = 0.0001 | — |

### What this does **not** claim

Kept here, in the open, rather than in a footnote:

- 🔸 **Only one of three explainers has run.** `llm-grounded` and `llm-free` are
  empty pending an API key, and `paper/results.md` says so *at the top* rather
  than presenting a one-explainer table as a comparison.
- 🔸 **The retrieval-quality table is partly circular.** Relevance is a
  citation-based proxy, and PPR propagates along those same citation edges. The
  `rrf+ppr` gap is inflated by construction. Nothing in the faithfulness
  analysis depends on it.
- 🔸 **Prerequisite rule 3 is inert.** No API serves section-segmented full text,
  so trees rest on rules 1 and 2 only. The rule is implemented, not exercised.

---

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

---

## Layout

```
synapse/
  ppr.py         Personalized PageRank with LIVE per-arc flow recording
  graph.py       Immutable typed graph; ablation returns new graphs
  retrieve.py    BM25 + dense + RRF + PPR, per-signal attribution
  explain.py     Three explainers, differing only in what they may see
  ablate.py      The contribution: control first, then necessity/sufficiency
  metrics.py     Bootstrap CIs and paired permutation tests
  tree.py        Prerequisite DAG derivation
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

Data flows **one direction**: `ingest → db → graph → retrieve → explain → ablate → artifact → GUI`.
The GUI recomputes nothing. Every number on screen came from the harness.

---

## The one interesting engineering decision

<details open>
<summary><b>Why PPR is written here instead of imported from networkx</b></summary>

<br>

`networkx.pagerank` returns node scores and nothing else. But this entire
experiment rests on knowing **which edges delivered mass to a candidate** — and a
score vector cannot answer that.

Reconstructing edge contributions afterwards would be a *model* of what the
propagation did, not a *record* of it. The ablation would then be testing the
reconstruction's assumptions rather than the ranker's behaviour, and no reviewer
could tell the two apart from the output.

So propagation is ~80 lines of power iteration, with per-arc flow recorded
**while the iteration runs**. At the fixed point, the recorded flows satisfy,
exactly:

```
x(v) = teleport(v) + Σ over in-arcs a of flow(a)
```

The residual of this identity is written into every run artifact
(**observed: 0.0**) and asserted in the test suite against an independent
naive-loop reference at `1e-12` tolerance.

**Attribution is arithmetic over recorded quantities, not inference about them.**

*Cost of the decision: ~170 lines to own and test, and one maintained dependency
given up.* Recorded in full in [docs/BUILD-RECORD.md](docs/BUILD-RECORD.md) §3.
</details>

---

## Guardrails enforced in code

Not in a style guide — in the source, where they cannot be forgotten.

| Guardrail | Why it is not optional |
|---|---|
| **The random control runs on every cell** | The null under test is exchangeability. Excluding cited edges from the control pool would test a weaker null and bias the protocol toward whichever explainer cites least. |
| **`llm-free` never sees the graph** | Asserted *mechanically* — a test scans the prompt for every edge id, node id, author name, venue name, and meta-path phrase in the corpus. |
| **Empty explanations are `untestable`** | Scoring them zero would launder an empty explanation into a harmless one. |
| **Tree edges come from `tree.py`, never an LLM** | An LLM-authored tree is unfalsifiable, and cannot be ablated. |
| **No mean without spread** | `Distribution` carries its own bootstrap CI, so no code path *can* print a centre without one. |
| **Trial-count sanity check** | With *n* control trials the smallest achievable *p* is 1/(n+1). The harness refuses to start if that floor is not below α. |

---

## Determinism

Fixed seed. Every API and LLM response cached to sqlite — **including 404s and
backoff jitter**. Every artifact embeds the resolved config, a corpus edge
digest, and the code revision.

A replay is deterministic and fully offline.

---

## Tests

```bash
.venv/bin/python -m pytest -q --cov=synapse
```

**215 tests · 85% coverage.**

The strictest live in `tests/test_ppr.py` (flow exactness) and
`tests/test_ablate.py`, whose decisive case checks the harness can separate a
candidate whose rank is *genuinely caused* by citation structure from one that
ranks on text while merely sharing a crowded venue.

`tests/test_gui_smoke.py` drives a real Chromium and asserts the WCAG floor:
arrow-key traversal, visible focus rings, `prefers-reduced-motion`, and that no
node state is conveyed by colour alone. It also asserts that an **unfaithful
verdict renders at exactly the same size as a faithful one** — because a
verdict shrunk by its own UI is a verdict quietly withdrawn.

---

<div align="center">

**The contribution is the protocol, not the ranking.**

</div>
