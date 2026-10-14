"""Ingest stage: manifest -> parsed markdown -> chunks on disk.

Layout under ``corpus.processed_dir`` (all gitignored; CLAUDE.md rule 2):

    parsed/<parser>/<doc_id>.md            parsed markdown
    parsed/<parser>/<doc_id>.meta.json     title, sections, parse stats, warnings
    chunks/<fingerprint>/chunks.jsonl      one chunk per line
    chunks/<fingerprint>/chunkset.json     the knobs that produced them

``<fingerprint>`` is ``PipelineConfig.ingest_fingerprint()``, so two ablation
rows with different chunking knobs keep separate chunk sets instead of one
overwriting the other.

The manifest's ``status`` is only advanced to ``ingested`` with
``--update-manifest``: a routine re-run should not dirty a committed file.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from quant_rag.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from quant_rag.ingest.chunk import Chunk, build_chunker
from quant_rag.ingest.manifest import (
    Document,
    iter_documents,
    load_manifest,
    save_manifest,
)
from quant_rag.ingest.parse import ParsedDoc, parse_document
from quant_rag.ingest.tokenizer import get_tokenizer


@dataclass
class DocReport:
    doc_id: str
    outcome: str  # parsed | error
    n_chars: int = 0
    n_sections: int = 0
    n_chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class IngestReport:
    fingerprint: str
    tokenizer: str
    tokenizer_approximate: bool
    docs: list[DocReport] = field(default_factory=list)
    parsed_dir: Path | None = None
    chunks_path: Path | None = None

    @property
    def failed(self) -> list[DocReport]:
        return [d for d in self.docs if d.outcome == "error"]

    @property
    def flagged(self) -> list[DocReport]:
        return [d for d in self.docs if d.warnings]

    @property
    def n_chunks(self) -> int:
        return sum(d.n_chunks for d in self.docs)


def write_parsed(parsed: ParsedDoc, parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{parsed.doc_id}.md").write_text(parsed.text, encoding="utf-8")
    (parsed_dir / f"{parsed.doc_id}.meta.json").write_text(
        json.dumps(parsed.to_meta(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def ingest_corpus(
    config: PipelineConfig,
    *,
    only: set[str] | None = None,
    update_manifest: bool = False,
    log: Callable[[str], None] = print,
) -> IngestReport:
    manifest_path = config.resolve(config.corpus.manifest)
    raw_dir = config.resolve(config.corpus.raw_dir)
    processed_dir = config.resolve(config.corpus.processed_dir)
    chunking = config.ingest.chunking

    manifest = load_manifest(manifest_path)
    tokenizer = get_tokenizer(chunking.tokenizer)
    chunker = build_chunker(chunking, tokenizer)
    fingerprint = config.ingest_fingerprint()

    parsed_dir = processed_dir / "parsed" / config.ingest.parser
    chunks_path = processed_dir / "chunks" / fingerprint / "chunks.jsonl"
    report = IngestReport(
        fingerprint=fingerprint,
        tokenizer=tokenizer.name,
        tokenizer_approximate=tokenizer.approximate,
        parsed_dir=parsed_dir,
        chunks_path=chunks_path,
    )

    if tokenizer.approximate:
        log(
            f"tokenizer: {tokenizer.name} (APPROXIMATE -- "
            f"{chunking.tokenizer} unavailable; token counts are estimates)"
        )
    else:
        log(f"tokenizer: {tokenizer.name}")
    log(
        f"chunking: strategy={chunking.strategy} chunk_tokens={chunking.chunk_tokens} "
        f"overlap_tokens={chunking.overlap_tokens} fingerprint={fingerprint}"
    )

    documents: list[Document] = [
        doc for doc in iter_documents(manifest) if only is None or doc["doc_id"] in only
    ]
    if only:
        for doc_id in sorted(only - {doc["doc_id"] for doc in documents}):
            report.docs.append(DocReport(doc_id, "error", detail="not in manifest"))
            log(f"[error]  {doc_id}: not in manifest")

    all_chunks: list[Chunk] = []
    ingested: list[Document] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        try:
            parsed = parse_document(doc, raw_dir, parser=config.ingest.parser)
        except (FileNotFoundError, NotImplementedError, OSError, RuntimeError) as exc:
            report.docs.append(DocReport(doc_id, "error", detail=str(exc)))
            log(f"[error]  {doc_id}: {exc}")
            continue

        write_parsed(parsed, parsed_dir)
        chunks = chunker.chunk(parsed)
        all_chunks.extend(chunks)
        ingested.append(doc)

        report.docs.append(
            DocReport(
                doc_id=doc_id,
                outcome="parsed",
                n_chars=len(parsed.text),
                n_sections=parsed.stats["n_sections"],
                n_chunks=len(chunks),
                warnings=parsed.warnings,
            )
        )
        flag = " FLAGGED" if parsed.warnings else ""
        log(
            f"[ok]     {doc_id}: {len(parsed.text):>7} chars  "
            f"{parsed.stats['n_sections']:>3} sections  {len(chunks):>4} chunks{flag}"
        )
        for warning in parsed.warnings:
            log(f"           ! {warning}")

    write_chunks(all_chunks, chunks_path)
    (chunks_path.parent / "chunkset.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "experiment": config.experiment.name,
                "config_file": str(config.source_path) if config.source_path else None,
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "parser": config.ingest.parser,
                "chunking": {
                    "strategy": chunking.strategy,
                    "chunk_tokens": chunking.chunk_tokens,
                    "overlap_tokens": chunking.overlap_tokens,
                    "tokenizer": chunking.tokenizer,
                },
                "tokenizer_used": tokenizer.name,
                "tokenizer_approximate": tokenizer.approximate,
                "n_docs": sum(1 for d in report.docs if d.outcome == "parsed"),
                "n_chunks": len(all_chunks),
                "flagged_docs": {d.doc_id: d.warnings for d in report.flagged},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if update_manifest and ingested:
        for doc in ingested:
            doc["status"] = "ingested"
        manifest["updated"] = datetime.now(UTC).strftime("%Y-%m-%d")
        save_manifest(manifest, manifest_path)
        log(f"manifest updated: {manifest_path}")

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", nargs="+", metavar="DOC_ID", help="restrict to these doc_ids")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="flip ingested documents' manifest status to 'ingested'",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = ingest_corpus(
        config,
        only=set(args.only) if args.only else None,
        update_manifest=args.update_manifest,
    )
    print(
        f"\nsummary: {sum(1 for d in report.docs if d.outcome == 'parsed')} docs parsed, "
        f"{report.n_chunks} chunks -> {report.chunks_path}"
    )
    if report.flagged:
        print(f"flagged parses ({len(report.flagged)}): review before trusting the index")
        for doc in report.flagged:
            print(f"  {doc.doc_id}: {'; '.join(doc.warnings)}")
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
