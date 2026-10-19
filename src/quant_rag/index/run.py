"""``quant-rag index``: embed the active chunk set into a vector index.

Re-running is a no-op by default. Embedding is the slow stage, and the index is
keyed by fingerprint, so an existing index that already verifies against the
current chunk set is reported and kept; ``--rebuild`` forces the work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_rag.config import DEFAULT_CONFIG_PATH, load_config
from quant_rag.index.store import (
    IndexMissingError,
    IndexPairingError,
    build_index,
    index_dir,
    load_index,
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-embed even if a matching index already exists",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        f"experiment: {config.experiment.name}  "
        f"ingest={config.ingest_fingerprint()} index={config.index_fingerprint()}"
    )

    if not args.rebuild:
        try:
            existing = load_index(config)
        except IndexMissingError:
            pass
        except IndexPairingError as exc:
            # Do not quietly rebuild over it: the owner should see that the
            # chunk set moved under an existing index before it is replaced.
            print(f"existing index does not match the current chunk set:\n{exc}")
            return 1
        else:
            print(
                f"index already built and verified: {index_dir(config)}\n"
                f"  {existing.meta.n_vectors} vectors, dim {existing.meta.dim}, "
                f"metric {existing.meta.metric}, built {existing.meta.built_at}\n"
                "pass --rebuild to re-embed"
            )
            return 0

    index, directory = build_index(config)
    meta = index.meta
    print(
        f"\nwrote {meta.n_vectors} vectors (dim {meta.dim}, metric {meta.metric}) "
        f"-> {directory}"
    )
    if meta.truncation["n_truncated"]:
        print(
            f"WARNING: {meta.truncation['n_truncated']}/{meta.truncation['n_texts']} "
            f"chunks were truncated to the model's "
            f"{meta.truncation['model_max_seq_length']}-token window, losing up to "
            f"{meta.truncation['max_tokens_dropped']} token(s) each. "
            f"ingest.chunking.chunk_tokens + 2 special tokens must fit in "
            f"{meta.truncation['model_max_seq_length']} to avoid this."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
