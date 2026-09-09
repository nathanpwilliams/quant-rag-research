"""``quant-rag`` command-line entry point."""

from __future__ import annotations

import argparse
import sys

from quant_rag import __version__
from quant_rag.ingest import download


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

    ask = sub.add_parser("ask", help="retrieve + answer a question over the corpus")
    ask.add_argument("question")
    ask.add_argument("--config", default="config/pipeline.yaml")
    ask.set_defaults(func=lambda args: _not_implemented("ask"))

    ev = sub.add_parser("eval", help="run the golden-set evaluation")
    ev.add_argument("--config", default="config/pipeline.yaml")
    ev.set_defaults(func=lambda args: _not_implemented("eval"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
