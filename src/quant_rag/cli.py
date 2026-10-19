"""``quant-rag`` command-line entry point."""

from __future__ import annotations

import argparse
import sys

from quant_rag import __version__
from quant_rag.index import run as index
from quant_rag.ingest import download
from quant_rag.ingest import run as ingest
from quant_rag.retrieve import ask as ask_cmd


def _not_implemented(stage: str) -> int:
    print(f"quant-rag {stage}: not implemented yet", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-rag")
    parser.add_argument("--version", action="version", version=f"quant-rag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="fetch arXiv PDFs listed in the manifest")
    download.add_arguments(dl)
    dl.set_defaults(func=download.run)

    ing = sub.add_parser("ingest", help="parse the corpus and build chunks")
    ingest.add_arguments(ing)
    ing.set_defaults(func=ingest.run)

    idx = sub.add_parser("index", help="embed the active chunk set into a vector index")
    index.add_arguments(idx)
    idx.set_defaults(func=index.run)

    ask = sub.add_parser("ask", help="retrieve chunks for a question (no generation yet)")
    ask_cmd.add_arguments(ask)
    ask.set_defaults(func=ask_cmd.run)

    ev = sub.add_parser("eval", help="run the golden-set evaluation")
    ev.add_argument("--config", default="config/pipeline.yaml")
    ev.set_defaults(func=lambda args: _not_implemented("eval"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
