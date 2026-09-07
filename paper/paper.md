# Does the Explanation Cause the Ranking? A Counterfactual Edge-Ablation Protocol for Scholarly Retrieval

**Target:** SDP / JCDL short / ECIR short — 4 pages.

> **Drafting rule for this file.** Sections 1–4 and 7 state the method and its
> limits and can be written now, because they describe what the instrument does
> rather than what it found. Sections 5 and 6 contain the numbers and the claim,
> and **§6 must not be written before `paper/results.md` is regenerated from a
> complete run.** The headline may be a null or negative finding; that is an
> acceptable outcome and the system must not be tuned to avoid it.
>
> Every number in §5 comes from `paper/results.md`, which is generated from
> `runs/*.json` by `paper/generate_results.py`. Do not retype numbers here.

---

## 1. Introduction

Scholarly retrieval systems report NDCG. Almost none report whether the reasons
they give for a ranking are the reasons that produced it.

This is not a cosmetic gap. Systems that surface *why* a paper was recommended —
"shares an author", "cited by the same work", "same research area" — present
those statements as accounts of the ranking. A reader treats them as causal. But
the explanation is typically produced by a separate module from the ranker, and
nothing in a standard evaluation checks that the two agree. An explanation can be
true (the shared author exists) and still be a non-explanation (the shared author
contributed nothing to the score).

We contribute a protocol, not a ranker. Given a system that ranks a candidate and
states an explanation naming specific graph edges, we delete exactly those edges,
recompute the ranking with identical parameters, and measure the displacement.
Against a count-matched random-edge control, this asks whether the stated reason
was necessary for the ranking it claims to explain.

**Contributions.**

1. A counterfactual edge-ablation protocol for retrieval explanations, with a
   random-edge control and a sufficiency test.
2. A Necessity Score that is comparable across candidates and list lengths.
3. A training-free testbed built for causal isolation, in which every ranking is
   reconstructible from persisted intermediate state.
4. A comparison of three explainers that differ only in what they are permitted
   to see.

## 2. Related work

*(To draft. The argument to make: faithfulness is a mature question in general
recommender systems and in GNN explainability — where counterfactual and
ablation-based evaluation of explanation subgraphs is standard — and is largely
absent in scholarly retrieval, where explanation quality is assessed by user
study or plausibility rating rather than by intervention. Position this work as
importing an established standard into a field that has not adopted it, not as
inventing a new one.)*

## 3. Protocol

Let a retrieval system rank candidate *c* for seed *s* at rank *r₀*, and emit an
explanation *E(s, c)* citing a set of graph edges *A*.

1. **Ablate.** Construct the graph *G \ A* and re-rank with identical parameters,
   giving *r₁*.
2. **Displacement.** *d = r₁ − r₀*.
3. **Necessity Score.** *N = (r₁ − r₀) / (|R| − r₀)*, where *|R|* is the ranked
   list length. This normalises by the room the candidate had to fall: a paper at
   rank 2 can drop 498 places, one at rank 480 can drop 20. Without it, averaging
   displacement systematically favours explanations attached to top-ranked items.
   *N = 1* means the candidate fell as far as it could. Negative values are
   reported, not clipped: they mean the cited edges were holding the candidate
   *down*.
4. **Random-edge control.** Draw *k = |A|* edges uniformly from the pool of edges
   incident to *s* or *c* — the same structural pool every explanation draws from
   — and repeat the ablation. Thirty trials per candidate give a per-candidate
   Monte Carlo p-value with a floor of 1/31 ≈ 0.032, below our α = 0.05.
   **The cited edges remain in the control pool.** The null under test is
   exchangeability ("these edges are no more special than any other *k*"), and a
   uniform draw is the correct null for it. Excluding them would test a weaker
   null and would become infeasible exactly when an explanation cites most of the
   neighbourhood, biasing the protocol toward whichever explainer cites least.
5. **Sufficiency.** Re-rank on the graph containing *only* *A*. Retention of 1.0
   means the cited edges alone reproduce the original rank.

A candidate is **faithful** iff its displacement exceeds its own control
distribution at *p* < 0.05. Explanations citing no edges are recorded as
*untestable* and excluded from the significance test rather than scored as zero
displacement — scoring them as zero would launder an empty explanation into a
harmless one.

## 4. Testbed

Deliberately training-free. Every component was chosen so that a causal chain can
be isolated, and where a simpler component made isolation easier it was preferred.

- **Corpus.** Citation neighbourhood expanded breadth-first from seed papers;
  typed nodes (paper, author, venue, concept) and typed edges (`cites`,
  `authored_by`, `published_in`, `shares_concept`).
- **Concepts.** YAKE keyphrases plus the source's own field labels. YAKE rather
  than KeyBERT because it is statistical and deterministic: a concept node's
  identity must not move when the dense retriever is swapped.
- **Retrieval.** BM25 + dense embeddings (all-MiniLM-L6-v2) fused by Reciprocal
  Rank Fusion (*k* = 60), combined with Personalized PageRank over the typed
  graph as a weighted sum of pool-normalised scores.
- **Provenance.** PPR is implemented directly rather than via a library, because
  the protocol needs to know *which edges delivered mass to a candidate* and a
  score vector cannot answer that. Per-arc flow is recorded during the final
  iteration, and at the fixed point satisfies exactly

  *x(v) = teleport(v) + Σ over in-arcs a of flow(a)*,

  so edge attribution is arithmetic over recorded quantities rather than a
  post-hoc reconstruction. The residual of this identity is written into every
  run artifact (observed: 0.0) and is asserted against an independent
  naive-loop reference implementation in the test suite.

**Explainers.** Three, differing only in what they may see:

| | Sees | Cites edges by |
|---|---|---|
| `metapath` | the graph only | deterministic typed traversal |
| `llm-grounded` | extracted meta-path facts only | returning fact indices |
| `llm-free` | titles and abstracts only, **no graph** | post-hoc grounding of claimed relations |

`llm-free`'s prose is graph-free; its claims are mapped onto edges only
afterwards, and claims corresponding to no real edge are counted as
`unrealised_claims` and cite nothing. That mapping is stated plainly because it is
the obvious reviewer question: the model's *reasoning* is graph-free, while its
*claims* are scored against the graph — which is the question being asked.

## 5. Results

See `results.md` (generated). **Do not retype numbers into this section**;
reference the tables and interpret them.

## 6. Finding

*(Do not write until §5 is complete for all three explainers across both corpora.
A null result — e.g. that no explainer beats its control, or that the
deterministic explainer is not distinguishable from the LLM ones — is a
publishable outcome for this paper and must be reported as found.)*

## 7. Limitations

State these plainly; several are design choices, not defects.

1. **The testbed is not state-of-the-art, by design.** No trained GNN, no
   contrastive alignment, no learned reranker. Every learned component is a
   confound for a causal claim about edges. Low ranking quality does not weaken
   the protocol; the instrument is the contribution.
2. **The NDCG table is partly circular.** Relevance is a citation-based proxy and
   PPR propagates along citation edges, so the graph configuration is scored
   against a target built from its own signal. Reported only to show the
   configurations are distinguishable. No faithfulness result depends on it.
3. **Prerequisite rule 3 is inert.** Related-work-section membership requires
   section-segmented full text, which neither OpenAlex nor the S2 Graph API
   serves. Trees rest on rules 1 and 2 only.
4. **Cycle-breaking discards real edges.** Citation graphs contain cycles
   (preprint/published pairs, simultaneous submissions). Every dropped edge is
   logged into the tree artifact and reported in `results.md`.
5. **Corpus provenance.** The primary corpus was built from OpenAlex rather than
   Semantic Scholar: the unauthenticated S2 Graph API returned HTTP 429 on every
   request from our host during the build window. Both sources normalise to an
   identical schema and identical typed edges, and each run artifact records
   which source produced its corpus. Results generalise to the neighbourhood
   sampled, not to scholarly retrieval at large.
6. **Explanation-to-edge mapping for the free-form explainer** is grounded by
   relation type rather than by matching free-text detail against node labels.
   Matching on detail would let a vague claim collect every edge of that type and
   inflate apparent faithfulness.
7. **Single ranker.** The protocol is demonstrated on one retrieval architecture.
   Whether these faithfulness rates transfer to learned rankers is untested, and
   the ablation is only meaningful for systems whose explanations name graph
   structure.

## 8. Reproducibility

```bash
synapse build  --seeds data/seeds.txt --hops 2      # corpus + typed graph
synapse ablate <seed> --explainer all               # the experiment
synapse eval   --dataset primary --report           # the results table
python paper/generate_results.py                    # regenerate results.md
```

Every run writes a JSON artifact to `runs/` embedding the resolved config, a
corpus edge digest, and the code revision. Fixed seed; all API and LLM responses
cached to sqlite, so a replay is deterministic and offline.
