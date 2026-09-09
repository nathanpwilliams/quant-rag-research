"""Load, validate, and save ``data/raw/manifest.json``.

The manifest is the canonical registry of ``doc_id`` values referenced by
``eval/golden_set.jsonl``. Documents are kept as plain dicts so unknown keys
(and key order) round-trip untouched; tooling only ever *adds* keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.json"
DEFAULT_RAW_DIR = DEFAULT_MANIFEST_PATH.parent

SOURCE_TYPES = frozenset({"arxiv", "web", "note"})
STATUSES = frozenset({"to_download", "to_write", "downloaded", "ingested"})
REQUIRED_DOC_KEYS = (
    "doc_id",
    "title",
    "authors",
    "year",
    "source_type",
    "arxiv_id",
    "url",
    "tags",
    "role",
    "status",
)

Manifest = dict[str, Any]
Document = dict[str, Any]


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    """Write with the same formatting the committed file uses (2-space, UTF-8)."""
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def iter_documents(
    manifest: Manifest,
    *,
    source_type: str | None = None,
    status: str | None = None,
) -> Iterator[Document]:
    for doc in manifest.get("documents", []):
        if source_type is not None and doc.get("source_type") != source_type:
            continue
        if status is not None and doc.get("status") != status:
            continue
        yield doc


def validate_manifest(manifest: Manifest) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    errors: list[str] = []
    for key in ("corpus_version", "updated", "documents"):
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")
    docs = manifest.get("documents", [])
    if not isinstance(docs, list):
        return errors + ["'documents' must be a list"]

    seen: set[str] = set()
    for i, doc in enumerate(docs):
        label = doc.get("doc_id", f"<index {i}>")
        for key in REQUIRED_DOC_KEYS:
            if key not in doc:
                errors.append(f"{label}: missing key {key!r}")
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            errors.append(f"{label}: doc_id must be a non-empty string")
            continue
        if doc_id in seen:
            errors.append(f"{doc_id}: duplicate doc_id")
        seen.add(doc_id)

        source_type = doc.get("source_type")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{doc_id}: unknown source_type {source_type!r}")
        elif not doc_id.startswith(f"{source_type}_"):
            errors.append(f"{doc_id}: doc_id prefix does not match source_type {source_type!r}")

        if doc.get("status") not in STATUSES:
            errors.append(f"{doc_id}: unknown status {doc.get('status')!r}")

        arxiv_id = doc.get("arxiv_id")
        if source_type == "arxiv":
            if not arxiv_id:
                errors.append(f"{doc_id}: arxiv source needs arxiv_id")
            elif doc_id != f"arxiv_{arxiv_id}":
                errors.append(f"{doc_id}: doc_id must equal 'arxiv_' + arxiv_id ({arxiv_id})")
        elif arxiv_id is not None:
            errors.append(f"{doc_id}: non-arxiv source must have arxiv_id null")

        if not isinstance(doc.get("tags"), list):
            errors.append(f"{doc_id}: tags must be a list")
    return errors
