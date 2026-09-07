# Results

<!-- GENERATED FILE — do not edit by hand.
     Regenerate with: python paper/generate_results.py
     Every number below is read from a run artifact in runs/. -->

Artifacts: 1 ablation run(s), 1 evaluation run(s), 1 tree(s).

> **Incomplete coverage.** These explainers were not run and are absent from every table below: `llm-free`, `llm-grounded`. They require `OPENROUTER_API_KEY`. No result here should be read as a comparison across all three explainers.

## Table 1 — Corpora

| source | papers | authors | venues | concepts | cites | shares_concept | edge digest |
|---|---|---|---|---|---|---|---|
| openalex:500 | 500 | 1821 | 119 | 478 | 1452 | 2872 | 928aaea49117d64d |


## Table 2 — Retrieval quality

> **This table is partly circular and must not be read as evidence that the graph signal is better.** Relevance here is a citation-based proxy (papers the seed cites, and papers citing the seed) because no human judgements exist for this corpus. Personalized PageRank propagates along those same citation edges, so the `rrf+ppr` row is scored against a target its own signal is built from. The gap between `rrf+ppr` and `rrf-only` is therefore inflated by construction and is reported only to show the retrieval configurations are distinguishable, not to claim ranking quality. Nothing in the faithfulness analysis depends on this table.

| corpus | signals | NDCG@10 [95% CI] | MRR [95% CI] |
|---|---|---|---|
| primary | bm25-only | 0.201 [0.083, 0.338] | 0.417 [0.183, 0.733] |
| primary | dense-only | 0.240 [0.106, 0.373] | 0.312 [0.171, 0.433] |
| primary | rrf-only | 0.237 [0.102, 0.371] | 0.373 [0.280, 0.467] |
| primary | rrf+ppr | 0.590 [0.551, 0.636] | 0.800 [0.600, 1.000] |


## Table 3 — Explanation faithfulness, per seed

Displacement is the change in rank after deleting exactly the edges the explanation cited. The random control deletes the same NUMBER of edges drawn from the same incident pool. An explanation is faithful only where its displacement beats its own control distribution at p < 0.05.

| seed | explainer | n | displacement [95% CI] | random control [95% CI] | necessity [95% CI] | % faithful | p |
|---|---|---|---|---|---|---|---|
| W3015883388 | metapath | 20 | 14.60 [11.20, 18.10] | 2.42 [1.52, 3.35] | 0.030 [0.023, 0.037] | 75% | 0.0001 |


## Table 4 — Explanation faithfulness, aggregated

| corpus | explainer | n | necessity [95% CI] | control [95% CI] | % faithful | p | empty expl. | unrealised claims |
|---|---|---|---|---|---|---|---|---|
| primary | metapath | 100 | 0.028 [0.023, 0.034] | 0.003 [0.002, 0.004] | 58% | 0.0001 | 0 | 0 |


## Limitations recorded by the harness

Cycle-breaking edge drops across 1 tree(s): **0**.

No cycles were encountered in the current trees. This is a property of these particular neighbourhoods, not a guarantee: citation graphs contain cycles (preprint/published pairs, simultaneous submissions) and the harness logs every drop when they occur.

Prerequisite rule 3 (related-work membership) contributed **no edges** in 1 of 1 tree(s): neither OpenAlex nor the Semantic Scholar Graph API serves section-segmented full text, so the rule is implemented but inert. Trees here rest on rules 1 and 2 only.

