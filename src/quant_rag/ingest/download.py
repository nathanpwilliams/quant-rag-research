"""Download the arXiv PDFs listed in the manifest into ``data/raw/``.

Deliberately stdlib-only: corpus assembly must work before any of the
pipeline's heavyweight dependencies are installed.

For every document with ``source_type == "arxiv"`` and ``status == "to_download"``:

1. Query the arXiv API for the paper's metadata (title, latest version).
2. Compare the returned title with the manifest title. The manifest flags some
   IDs as "from memory"; a mismatch almost certainly means the ``arxiv_id`` is
   wrong, so the paper is *not* downloaded unless ``--accept-title-mismatch``.
3. Download the version-pinned PDF to ``data/raw/<doc_id>.pdf``.
4. Record ``file``, ``sha256``, ``arxiv_version``, ``arxiv_title`` and
   ``downloaded_at`` on the manifest entry and flip ``status`` to ``downloaded``.

arXiv's terms of use ask for at least 3 seconds between requests:
https://info.arxiv.org/help/api/tou.html
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from quant_rag.ingest.manifest import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    REPO_ROOT,
    Document,
    iter_documents,
    load_manifest,
    save_manifest,
)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "quant-rag-research/0.1 (corpus downloader; +https://github.com/)"
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_MIN_TITLE_SIMILARITY = 0.85
HTTP_TIMEOUT_SECONDS = 60

HttpGet = Callable[[str], bytes]


class ArxivNotFound(Exception):
    """The arXiv API returned no entry for the requested id."""


@dataclass(frozen=True)
class ArxivMeta:
    arxiv_id: str
    title: str
    version: int
    published: str


@dataclass
class DocResult:
    doc_id: str
    outcome: str  # downloaded | already_present | skipped | title_mismatch | not_found | error
    detail: str = ""
    meta: ArxivMeta | None = None


@dataclass
class Report:
    results: list[DocResult] = field(default_factory=list)

    @property
    def failed(self) -> list[DocResult]:
        return [r for r in self.results if r.outcome in {"title_mismatch", "not_found", "error"}]

    def add(self, result: DocResult) -> DocResult:
        self.results.append(result)
        return result


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no I/O)
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/LaTeX noise, collapse whitespace."""
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def pdf_url(arxiv_id: str, version: int | None = None) -> str:
    suffix = f"v{version}" if version else ""
    return f"{ARXIV_PDF}/{arxiv_id}{suffix}"


def api_url(arxiv_id: str) -> str:
    return f"{ARXIV_API}?{urllib.parse.urlencode({'id_list': arxiv_id, 'max_results': 1})}"


def parse_atom(xml_bytes: bytes, arxiv_id: str) -> ArxivMeta:
    """Extract title/version from an arXiv API Atom feed for a single id."""
    root = ET.fromstring(xml_bytes)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise ArxivNotFound(f"no entry returned for {arxiv_id}")
    entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
    title = " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split())
    if "api/errors" in entry_id or title.lower() == "error":
        summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
        raise ArxivNotFound(f"arXiv API error for {arxiv_id}: {summary.strip()}")
    match = re.search(r"/abs/(.+?)v(\d+)$", entry_id)
    if not match:
        raise ArxivNotFound(f"could not parse entry id {entry_id!r} for {arxiv_id}")
    returned_id, version = match.group(1), int(match.group(2))
    if returned_id != arxiv_id:
        raise ArxivNotFound(f"asked for {arxiv_id}, API returned {returned_id}")
    published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
    return ArxivMeta(arxiv_id=arxiv_id, title=title, version=version, published=published)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(doc: Document, raw_dir: Path) -> Path:
    return raw_dir / f"{doc['doc_id']}.pdf"


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


class _Throttle:
    """Enforce a minimum gap between consecutive network calls."""

    def __init__(self, delay: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self.delay = delay
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None and self.delay > 0:
            remaining = self.delay - (time.monotonic() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = time.monotonic()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def download_corpus(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    only: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    accept_title_mismatch: bool = False,
    min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
    delay: float = DEFAULT_DELAY_SECONDS,
    fetch: HttpGet = http_get,
    log: Callable[[str], None] = print,
) -> Report:
    manifest = load_manifest(manifest_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    throttle = _Throttle(delay)
    report = Report()
    changed = False

    candidates = [
        doc
        for doc in iter_documents(manifest, source_type="arxiv")
        if (only is None or doc["doc_id"] in only)
        and (force or doc.get("status") == "to_download")
    ]
    if only:
        missing = only - {doc["doc_id"] for doc in iter_documents(manifest, source_type="arxiv")}
        for doc_id in sorted(missing):
            report.add(DocResult(doc_id, "error", "not an arxiv doc_id in the manifest"))

    for doc in candidates:
        doc_id, arxiv_id = doc["doc_id"], doc["arxiv_id"]
        dest = target_path(doc, raw_dir)

        if dest.exists() and not force:
            report.add(DocResult(doc_id, "already_present", str(dest)))
            log(f"[present]  {doc_id}: {dest.name} exists (status={doc['status']})")
            continue

        # 1. metadata
        try:
            throttle.wait()
            meta = parse_atom(fetch(api_url(arxiv_id)), arxiv_id)
        except ArxivNotFound as exc:
            report.add(DocResult(doc_id, "not_found", str(exc)))
            log(f"[404]      {doc_id}: {exc}")
            continue
        except (urllib.error.URLError, OSError, ET.ParseError) as exc:
            report.add(DocResult(doc_id, "error", f"metadata fetch failed: {exc}"))
            log(f"[error]    {doc_id}: metadata fetch failed: {exc}")
            continue

        # 2. title check
        similarity = title_similarity(doc["title"], meta.title)
        if similarity < min_title_similarity and not accept_title_mismatch:
            detail = f"similarity={similarity:.2f} manifest={doc['title']!r} arxiv={meta.title!r}"
            report.add(DocResult(doc_id, "title_mismatch", detail, meta))
            log(f"[mismatch] {doc_id}: {detail}")
            continue

        if dry_run:
            report.add(DocResult(doc_id, "skipped", "dry run", meta))
            log(f"[dry-run]  {doc_id}: v{meta.version} sim={similarity:.2f} {meta.title!r}")
            continue

        # 3. pdf
        try:
            throttle.wait()
            payload = fetch(pdf_url(arxiv_id, meta.version))
        except (urllib.error.URLError, OSError) as exc:
            report.add(DocResult(doc_id, "error", f"pdf fetch failed: {exc}", meta))
            log(f"[error]    {doc_id}: pdf fetch failed: {exc}")
            continue
        if not payload.startswith(b"%PDF"):
            report.add(DocResult(doc_id, "error", "response is not a PDF", meta))
            log(f"[error]    {doc_id}: response is not a PDF (got {payload[:16]!r})")
            continue
        tmp = dest.with_suffix(".pdf.part")
        tmp.write_bytes(payload)
        tmp.replace(dest)

        # 4. bookkeeping
        try:
            rel = str(dest.resolve().relative_to(REPO_ROOT))
        except ValueError:
            rel = str(dest)
        doc.update(
            {
                "status": "downloaded",
                "file": rel,
                "sha256": sha256_bytes(payload),
                "arxiv_version": meta.version,
                "arxiv_title": meta.title,
                "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            }
        )
        changed = True
        report.add(DocResult(doc_id, "downloaded", f"v{meta.version} {len(payload)} bytes", meta))
        log(f"[ok]       {doc_id}: v{meta.version} {len(payload) / 1e6:.1f} MB sim={similarity:.2f}")

    if changed:
        manifest["updated"] = datetime.now(UTC).strftime("%Y-%m-%d")
        save_manifest(manifest, manifest_path)
        log(f"manifest updated: {manifest_path}")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--only", nargs="+", metavar="DOC_ID", help="restrict to these doc_ids"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present / not to_download"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="query metadata and check titles; write nothing"
    )
    parser.add_argument(
        "--accept-title-mismatch",
        action="store_true",
        help="download even when the arXiv title differs from the manifest title",
    )
    parser.add_argument(
        "--min-title-similarity", type=float, default=DEFAULT_MIN_TITLE_SIMILARITY
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="seconds between requests"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    report = download_corpus(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        only=set(args.only) if args.only else None,
        force=args.force,
        dry_run=args.dry_run,
        accept_title_mismatch=args.accept_title_mismatch,
        min_title_similarity=args.min_title_similarity,
        delay=args.delay,
    )
    counts: dict[str, int] = {}
    for result in report.results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do"
    print(f"summary: {summary}")
    if report.failed:
        print(
            "fix the arxiv_id in the manifest for mismatched/not-found entries, "
            "or rerun with --accept-title-mismatch if the arXiv title is just a rename.",
            file=sys.stderr,
        )
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
