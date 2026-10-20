# quant-rag-research

**Status (Oct 2026): in progress.** Corpus ingestion is complete — 18
provenance-verified sources (SHA-256 recorded for each; the 16 arXiv papers
title-checked against the arXiv API), PyMuPDF parsing, and config-driven
chunking (fixed and structure-aware), yielding 459 chunks with no parse-quality
flags. Indexing and dense retrieval work end to end: BGE embeddings over the
active chunk set, exact cosine search, and the `quant-rag index` / `quant-rag
ask` commands, with 172 passing tests. Planned: a 4-row retrieval ablation
(recall@k, MRR, faithfulness) against a human-annotated golden set.

Eval-centric RAG methodology study over a small RL / quant-finance paper corpus.
The deliverable is a measured ablation (fixed-chunk dense -> structure-aware ->
hybrid -> hybrid + rerank) with committed results, not a chat app.

## Setup

```bash
uv sync
uv run pytest
```

## Corpus assembly

`data/raw/manifest.json` is the canonical document registry. PDFs are gitignored;
fetch them with the stdlib-only downloader (arXiv rate limit respected):

```bash
uv run quant-rag download --dry-run   # verify titles/versions, write nothing
uv run quant-rag download             # fetch to data/raw/<doc_id>.pdf, update manifest
```

A manifest entry whose arXiv title does not match is skipped and reported; fix the
`arxiv_id` (or pass `--accept-title-mismatch` for a plain rename) and rerun.

## Indexing and retrieval

```bash
uv run quant-rag ingest                      # parse + chunk (writes a chunk set)
uv run quant-rag index                       # embed that chunk set (add --rebuild to force)
uv run quant-rag ask "What is the EIIE topology?"
```

`ask` prints the top chunks with doc_id, chunk_id, score, and a preview. There is
no answer generation yet.

## Golden set

`eval/golden_set.jsonl` is human-annotated by design and is never written by
tooling (CLAUDE.md rule 1). The validator is read-only and checks three layers —
JSONL/schema structure, cross-field coherence (an `answerable: true` row with no
gold evidence has no recall@k to compute), and corpus cross-reference (gold
`doc_id`s must name manifest documents, and locators must resolve to a chunk
under the *active* chunking config):

```bash
uv run quant-rag validate-golden-set              # schema + corpus cross-reference
uv run quant-rag validate-golden-set --strict     # warnings fail too
uv run quant-rag validate-golden-set --require-locators   # for CI once filled in
```

Exit status is 0 only when there are no errors, so it works as a pre-commit or
CI gate. Unfinished rows are marked with `TODO` placeholders and
`status: needs_corpus_fill`; the two must agree, and `status: ready` with a
`TODO` still in it is an error — that is the one mistake that puts a placeholder
into a measured run.

Locator resolution needs an ingested chunk set (gitignored), so it runs when one
is present and is skipped with a note otherwise; `--require-locators` turns that
skip into a failure so a green CI run cannot mean "the interesting half didn't
run".

## Design decisions

**`exact_numpy` is the default search backend, not FAISS.** The corpus is 459
chunks of 768 float32 — a 1.4 MB matrix. A query is one `(1,768) @ (768,459)`
matmul: ~350k multiply-adds, microseconds under BLAS, with no index structure to
build, tune, or persist. FAISS earns its complexity on approximate search over
millions of vectors; at this scale `IndexFlatIP` performs the identical
brute-force computation, just behind a native extension. `tests/test_backends.py`
asserts the two return the same top-k on corpus-shaped data, so the choice is an
implementation detail rather than an experimental variable.

It also avoids a hard crash rather than a hypothetical one. `faiss-cpu` and
`torch` each bundle their own `libomp.dylib` under different install names, so
macOS dyld loads two OpenMP runtimes and the process dies once both are used —
`OMP: Error #15` (SIGABRT), or SIGSEGV inside `__kmp_suspend_initialize_thread`
on an OpenMP worker thread once `KMP_DUPLICATE_LIB_OK` silences the abort. Since
`ask` must embed a query (torch) and search the index in one process, the default
path must not import faiss at all — and `exact_numpy` doesn't. `faiss_flat_ip`
remains selectable via `index.backend` for comparison and for a corpus that one
day needs ANN; it routes through `src/quant_rag/index/_runtime.py`, which
installs the OpenMP guard and documents its limits.

**Vectors are persisted backend-neutrally.** `index/<fingerprint>/vectors.npy`
plus `chunk_ids.json` and `index.json`; the backend is constructed from that
matrix at load time. So `index.backend` is not part of the index fingerprint and
switching backends never costs a re-embed.

**An index can never be paired with a chunk set it was not built from.** The
index fingerprint hashes the ingest fingerprint along with the embedding knobs,
so a different chunking config lands in a different directory. That alone is not
enough — the ingest fingerprint hashes the tokenizer *name*, so re-ingesting
after `transformers` became importable silently moves every chunk boundary under
an unchanged fingerprint. `index.json` therefore also records the SHA-256 of
`chunks.jsonl`, and loading re-hashes and refuses on mismatch.

**`chunk_tokens: 510`, because the window includes the special tokens.** BGE's
512-token window counts `[CLS]` and `[SEP]`, so the content budget is 512 - 2.
At 512 the encoder silently truncated 95% of chunk tails — doubly bad given
BGE pools on CLS, which already underweights tails.

Dropping to 510 left 2 chunks still overflowing by 1 token, for a separate
reason worth knowing: windowing counted tokens *in document context*, but the
encoder is handed the isolated slice. A window opening mid-word carries a
fragment, and WordPiece splits a fragment into more pieces alone than it
contributed inside the whole word — the tail of "statistic" is one token in
context and two as a bare "istic". `ingest.chunk.fit_to_budget` now measures the
slice that will actually be embedded and shrinks it to fit, so `n_tokens` is
truthful (it is recorded per chunk and read back by the eval) and the build
reports `no truncation`. `quant-rag index` still warns if that ever stops
holding.

**BGE's query instruction is a knob, not a constant.** The model card specifies
`"Represent this sentence for searching relevant passages: "` for queries and
none for passages, while also noting that v1.5 was trained to work without it,
loses only "a slight degradation" when omitted, and that the setting should be
chosen by measuring. `index.embedding.query_instruction` holds it; `""` is the
no-instruction ablation row. It never touches passages, so both settings share
one index.

See `CLAUDE.md` for layout, experiment rules, and the eval contract.
