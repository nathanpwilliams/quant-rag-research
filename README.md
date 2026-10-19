# quant-rag-research

**Status (Oct 2026): in progress.** Corpus ingestion is complete — 18
provenance-verified sources (SHA-256 recorded for each; the 16 arXiv papers
title-checked against the arXiv API), PyMuPDF parsing, and config-driven
chunking (fixed and structure-aware). Indexing and dense retrieval now work
end to end: BGE embeddings over the active chunk set, exact cosine search, and
the `quant-rag index` / `quant-rag ask` commands — 125 passing tests. Three
further own-notes documents are registered in the manifest but not yet written
(they currently index as empty templates). Planned: a 4-row retrieval ablation
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

## Design decisions

**`exact_numpy` is the default search backend, not FAISS.** The corpus is 461
chunks of 768 float32 — a 1.4 MB matrix. A query is one `(1,768) @ (768,461)`
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

**Chunks are truncated by 1-3 tokens and the build says so.** BGE's 512-token
window counts `[CLS]` and `[SEP]`, so `chunk_tokens: 512` of content does not
fit: 440 of 461 chunks lose up to 3 tokens off the end. The loss is tiny and at a
chunk tail, but it is measured, recorded in `index.json`, and printed as a
warning rather than left implicit. Setting `chunk_tokens: 510` would eliminate it
at the cost of a re-ingest and a new fingerprint.

**BGE's query instruction is a knob, not a constant.** The model card specifies
`"Represent this sentence for searching relevant passages: "` for queries and
none for passages, while also noting that v1.5 was trained to work without it,
loses only "a slight degradation" when omitted, and that the setting should be
chosen by measuring. `index.embedding.query_instruction` holds it; `""` is the
no-instruction ablation row. It never touches passages, so both settings share
one index.

See `CLAUDE.md` for layout, experiment rules, and the eval contract.
