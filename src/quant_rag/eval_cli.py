"""``quant-rag validate-golden-set``: check the golden set, change nothing.

Read-only by construction (CLAUDE.md rule 1). Exit status is 0 when there are
no errors, 1 otherwise, so this is usable as a pre-commit or CI gate; warnings
only fail under ``--strict``.

Locator resolution needs a chunk set on disk, which is gitignored, so it runs
when one is present and is skipped with a note when it is not. ``--require-
locators`` turns that skip into an error -- what CI should use once the corpus
is filled in, so a green run cannot mean "the interesting half did not run".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_rag.config import DEFAULT_CONFIG_PATH, load_config
from quant_rag.golden_set import ERROR, normalize_for_match, validate
from quant_rag.index.store import chunks_path, load_chunks
from quant_rag.ingest.manifest import iter_documents, load_manifest


def resolve_golden_set(config, explicit: Path | None) -> tuple[Path | None, str]:
    """Find the golden set, preferring an explicit path over the config's.

    The config names ``eval/golden_set.jsonl``, which is the filled-in file
    that does not exist yet; the committed template is
    ``eval/golden_set_seed.jsonl``. Rather than quietly validating whichever
    one happens to be there, say which one was used.
    """
    if explicit is not None:
        return (explicit if explicit.exists() else None), str(explicit)

    configured = config.resolve(config.eval.golden_set)
    if configured.exists():
        return configured, f"{configured} (from eval.golden_set)"

    seed = configured.with_name(configured.stem + "_seed" + configured.suffix)
    if seed.exists():
        return seed, (
            f"{seed} (fallback: eval.golden_set points at {configured.name}, "
            "which does not exist yet)"
        )
    return None, str(configured)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=None,
        help="file to validate (default: eval.golden_set from the config)",
    )
    parser.add_argument(
        "--require-locators",
        action="store_true",
        help="fail if no chunk set is available to resolve locators against",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures too"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the summary and any findings"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path, described = resolve_golden_set(config, args.golden_set)
    if path is None:
        print(f"golden set not found: {described}", file=sys.stderr)
        return 1

    manifest_doc_ids = frozenset(
        doc["doc_id"] for doc in iter_documents(load_manifest(config.resolve(config.corpus.manifest)))
    )

    chunk_bodies: dict[str, str] | None = None
    chunks_file = chunks_path(config)
    try:
        chunks = load_chunks(chunks_file)
    except Exception as exc:  # noqa: BLE001 - absent or unreadable: both mean "skip"
        if args.require_locators:
            print(
                f"--require-locators was given but no chunk set is available:\n{exc}",
                file=sys.stderr,
            )
            return 1
        chunk_note = f"skipped (no chunk set at {chunks_file}; run `quant-rag ingest`)"
    else:
        chunk_bodies = {c.chunk_id: normalize_for_match(c.body) for c in chunks}
        chunk_note = (
            f"{len(chunks)} chunks, ingest fingerprint {config.ingest_fingerprint()}"
        )

    report = validate(path, known_doc_ids=manifest_doc_ids, chunk_bodies=chunk_bodies)

    if not args.quiet:
        print(f"golden set: {described}")
        print(f"corpus:     {len(manifest_doc_ids)} manifest documents")
        print(f"locators:   {chunk_note}")
        print()

    for finding in report.findings:
        stream = sys.stderr if finding.severity == ERROR else sys.stdout
        print(finding, file=stream)
    if report.findings and not args.quiet:
        print()

    if not args.quiet:
        print(
            f"{report.n_rows} rows, {report.n_evidence} gold evidence entries"
        )
        if report.tiers:
            print("  tiers:    " + ", ".join(f"{k}={v}" for k, v in sorted(report.tiers.items())))
        if report.statuses:
            print(
                "  statuses: "
                + ", ".join(f"{k}={v}" for k, v in sorted(report.statuses.items()))
            )
        if report.locators_checked and report.locator_resolutions:
            resolved = sum(1 for n in report.locator_resolutions.values() if n)
            total = len(report.locator_resolutions)
            print(f"  locators: {resolved}/{total} filled-in locators resolve to a chunk")

    n_err, n_warn = len(report.errors), len(report.warnings)
    verdict = "FAIL" if n_err else ("FAIL (strict)" if n_warn and args.strict else "OK")
    print(f"{verdict}: {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err or (args.strict and n_warn) else 0


def main(argv: list[str] | None = None) -> int:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
