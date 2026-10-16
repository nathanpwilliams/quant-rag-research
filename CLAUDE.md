# quant-rag-research

Eval-centric RAG pipeline over an RL/quant-finance paper corpus. The deliverable is a
measured ablation (fixed-chunk dense -> structure-aware -> hybrid -> hybrid+rerank) with
committed results, not a chat app. "-research" suffix is intentional: this repo is a
methodology study, not a product.

## Architecture

PDF/markdown sources -> ingest (PyMuPDF parse + chunk) -> index (BGE embeddings in FAISS
+ BM25) -> retrieve (dense, sparse, RRF hybrid fusion, cross-encoder rerank) -> generate
(LLM answer w/ citations + abstention). Eval: eval/run_eval.py resolves golden-set
locators to chunk IDs under the ACTIVE chunking config, computes recall@k/MRR per tier,
then RAGAS metrics.

## Layout

- config/pipeline.yaml   — every experiment knob lives here; experiments = new config, never forked code
- data/raw/              — PDFs + manifest.json (PDFs gitignored; manifest committed)
- data/processed/        — parsed markdown + metadata (gitignored)
- src/quant_rag/         — ingest/ index/ retrieve/ generate/ pipeline.py
- eval/                  — golden_set.jsonl, run_eval.py, results/ (results CSVs are committed)
- tests/                 — pytest; chunking, fusion, parsing, locator-resolver

## Commands

- `uv sync` — install deps
- `uv run pytest` — run tests (run before every commit)
- `uv run quant-rag ask "..."` — retrieval CLI
- `uv run quant-rag eval --config config/pipeline.yaml` — full eval run

## Hard rules

1. NEVER edit `eval/golden_set.jsonl` answers, locators, or gold_evidence. It is
   human-annotated by design; a model filling it in invalidates the eval. Adding
   schema-level tooling (validation script) is fine; touching annotation content is not.
2. Never commit anything under `data/raw/` or `data/processed/` except
   `data/raw/manifest.json` and authored `note_*.md` documents (original written
   content, part of the reproducible corpus).
3. Config-driven experiments only: a new ablation row = a new YAML file in `config/`,
   never a code branch or commented-out block.
4. Eval results are append-only: new runs add CSVs under `eval/results/`; never
   overwrite or delete previous results.
5. Ask before adding any new dependency.

## Git discipline

- Small, atomic commits; one logical change each. Conventional-commit style:
  `feat(ingest): ...`, `fix(retrieve): ...`, `test: ...`, `docs: ...`, `eval: ...`.
- Run `uv run pytest` before committing; do not commit with failing tests.
- Never use `git push --force`, never amend or rebase pushed commits, never commit
  directly to a tag.
- Keep the tree clean: no stray scratch files, no `*_old.py`, no commented-out corpses.
  Scratch work goes in `notebooks/` or gets deleted.

## Division of labor

Claude Code handles scaffolding, boilerplate, tests, CI wiring, and refactors. The owner
hand-writes and must be able to defend `retrieve/` fusion+rerank logic and `eval/`
metric logic — for those modules, propose and review rather than bulk-generate, and flag
anything he should understand before it merges.
