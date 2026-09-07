# Corpora

Synapse builds its primary corpus from a live API and can additionally load two
offline evaluation corpora. All three normalise into the same schema and produce
the same typed edges, so nothing downstream can tell them apart except the
`source` column and the run artifact's corpus fingerprint.

## Primary corpus (live API)

```bash
synapse build --seeds data/seeds.txt --hops 2 --max-papers 500
```

### Source selection

`ingest_source` in `config.yaml` chooses the API. Override per-run with
`--source`.

| Source | Key needed | Status on this machine (2026-08-10) |
|---|---|---|
| `openalex` | none | **Working.** 500 papers in ~36 s. |
| `s2` | `S2_API_KEY` in practice | Unauthenticated pool returned HTTP 429 on every request after a short burst. |

**Why the default is OpenAlex.** The build prompt specifies Semantic Scholar. On
2026-08-10 the unauthenticated S2 Graph API rate-limited this host to the point
where a 500-paper expansion could not complete: an initial burst succeeded, then
every subsequent request returned 429 through ten retries with exponential
backoff to 60 s. OpenAlex serves the same graph structure — citations,
authorships, venues, topics — with an open rate limit and no key.

This is a data-availability fact, not a methodological change, but it is recorded
in every run artifact because corpus provenance changes what a result generalises
to. To use S2 instead:

```bash
export S2_API_KEY=...            # https://www.semanticscholar.org/product/api
synapse build --source s2 --seeds data/seeds.txt
```

### Seed file format

One id per line; `#` starts a comment. Accepted forms:

```
W3015883388                 OpenAlex work id (preferred — unambiguous)
arXiv:2004.12832            arXiv id
10.1145/3397271.3401075     DOI
title:Exact Paper Title     exact, case-insensitive title match
```

Prefer pinned OpenAlex ids. Resolving arXiv ids is unreliable — OpenAlex indexes
published versions under the publisher DOI, so DPR and SPECTER both 404 on their
arXiv DOI. A pinned id cannot silently resolve to the wrong paper; a title search
can, and seeding the wrong paper corrupts every downstream result.

### API quirks worth knowing

- **S2 `/references` and `/citations` reject dotted author subfields.** Passing
  `authors.authorId,authors.name` returns HTTP 400 (`Unrecognized or unsupported
  fields`) even though `/paper` accepts them. Use bare `authors`. This cost one
  silent build that produced 5 papers and zero citation edges while reporting
  success — hence `CorpusStats.expansion_failed`, which now makes a corpus with
  no `cites` edges a hard error.
- **OpenAlex `title` is frequently null**; `display_name` carries the title.
- **OpenAlex abstracts** ship as `abstract_inverted_index` and are reconstructed
  at ingest.

## Evaluation corpora (offline dumps)

Loaders are implemented for both formats but the dumps are large and are not
vendored. Both are read by `synapse.ingest.load_s2orc` / `load_dblp`, which
accept JSONL or a top-level JSON array and add citation edges only where both
endpoints are inside the loaded subset.

### S2ORC

Requires a Semantic Scholar API key and acceptance of the dataset terms.

1. Request access: <https://api.semanticscholar.org/datasets/v1/>
2. Download a slice of the `s2orc` or `papers` release.
3. Load a subset:

```python
from synapse.db import Database
from synapse.ingest import load_s2orc
load_s2orc(Database("data/synapse.sqlite"), "data/s2orc_subset.jsonl", max_papers=5000)
```

Expected fields: `corpusid`, `title`, `abstract`, `year`, `venue`, `authors`, and
either `citations`/`references` id lists or `outbound_citations`.

### DBLP-citation-v14

From ArnetMiner: <https://www.aminer.org/citation> (~5 GB uncompressed).

Slice it before loading — the loader streams JSONL but a top-level JSON array is
read whole:

```bash
head -c 200000000 dblp.v14.json > data/dblp_subset.json
```

```python
from synapse.ingest import load_dblp
load_dblp(Database("data/synapse.sqlite"), "data/dblp_subset.json", max_papers=5000)
```

Expected fields: `id`, `title`, `abstract`, `year`, `venue` (object or string),
`authors`, `references`, `n_citation`.

Then:

```bash
synapse eval --dataset s2orc --report
synapse eval --dataset dblp --report
```

## Caching

Every API response is cached in sqlite keyed by request identity, including 404s.
A cached record is never re-fetched. An interrupted build resumes for free, and a
replay months later reproduces the same graph offline. Deleting
`data/synapse.sqlite` discards the cache and forces a full refetch — which, on
the unauthenticated S2 pool, may not complete.
