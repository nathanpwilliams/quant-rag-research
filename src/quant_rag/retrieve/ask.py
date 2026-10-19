"""``quant-rag ask``: retrieve and print chunks. No generation yet.

Printing retrieval on its own is the point at this stage: the ablation is about
what reaches the generator, and a wrong chunk is much easier to see here than
inside a fluent answer. ``generate/`` lands later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_rag.config import DEFAULT_CONFIG_PATH, load_config
from quant_rag.index.store import IndexStoreError
from quant_rag.retrieve.dense import Hit, build_retriever

PREVIEW_CHARS = 200


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """First ``limit`` characters on one line, so hits stay scannable."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def format_hit(hit: Hit) -> str:
    chunk = hit.chunk
    where = f" [{chunk.section_title}]" if chunk.section_title else ""
    page = f" p{chunk.page_start}" if chunk.page_start is not None else ""
    return (
        f"{hit.rank}. {hit.score:+.4f}  {chunk.doc_id}  {chunk.chunk_id}{page}{where}\n"
        f"   {preview(chunk.body)}"
    )


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="how many chunks to print (default 5; retrieval depth is retrieve.top_k)",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        retriever = build_retriever(config)
    except IndexStoreError as exc:
        print(exc, file=sys.stderr)
        return 1

    # Retrieve to the config's depth, then print the requested slice, so what
    # is shown is a window onto the same ranking the eval scores.
    top_k = config.retrieve.top_k
    hits = retriever.search(args.question, top_k)

    print(f"query: {args.question}")
    print(
        f"mode={retriever.mode} metric={retriever.metric} top_k={top_k} "
        f"index={config.index_fingerprint()} "
        f"query_instruction={'yes' if config.index.embedding.query_instruction else 'no'}"
    )
    print()
    if not hits:
        print("no hits")
        return 0
    for hit in hits[: args.k]:
        print(format_hit(hit))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
