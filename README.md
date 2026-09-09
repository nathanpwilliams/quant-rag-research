# quant-rag-research

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

See `CLAUDE.md` for layout, experiment rules, and the eval contract.
