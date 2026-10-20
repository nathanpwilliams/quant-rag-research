"""Schema validation for ``eval/golden_set.jsonl``.

The golden set is human-annotated by design (CLAUDE.md rule 1), so nothing here
ever writes to it. This module only reads it and reports what is wrong, which
rule 1 explicitly permits: schema-level tooling is fine, annotation content is
the owner's.

What it checks, in three layers:

1. **Structure.** Valid JSONL, exactly the known keys, right types, unique ids.
   Catches the ordinary damage of hand-editing a 10-row JSONL file: a trailing
   comma, a key typo'd to ``gold_evidenc``, a row pasted twice.
2. **Cross-field coherence.** The invariants that make a row *scorable* rather
   than merely well-formed. An ``answerable: true`` row with no gold evidence
   has no recall@k to compute; an ``answerable: false`` row *with* gold
   evidence contradicts itself. These would not crash a runner -- they would
   silently produce a meaningless number, which is worse.
3. **Corpus cross-reference.** Gold ``doc_id``s must name real manifest
   documents, and (optionally, when a chunk set is on disk) locators must
   actually resolve to a chunk. An annotation that points at a document that
   left the corpus, or quotes a phrase the parser never produced, is gold
   evidence that can never be retrieved -- a permanent zero on that row that
   looks like a retrieval failure.

The ``TODO`` convention
-----------------------

Unfilled annotations are marked with ``TODO`` (``TODO_ppo_paper_1`` as a
doc_id, ``"TODO: reward definition phrase"`` as a locator), and ``status``
records whether a row is finished. The two must agree: ``status: ready`` with a
``TODO`` left in it is the single most costly mistake available here, because it
is the one that puts a placeholder into a measured run. That pairing is an
error; the reverse (no ``TODO`` left but still flagged unfinished) is a warning,
since it is the harmless half and may just be an edit in progress.

Boundary with ``eval/run_eval.py``
----------------------------------

``locator_matches`` answers a validation question -- "is this annotation usable
at all?" -- and is deliberately simple. The authoritative locator resolver used
for *scoring* belongs to the eval's metric logic, which is the owner's to write
(CLAUDE.md division of labor). If ``run_eval.py`` wants to reuse this matching,
it should do so explicitly rather than by accident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TIERS = frozenset({"lookup", "synthesis", "vocab_mismatch", "unanswerable"})
STATUSES = frozenset({"ready", "needs_corpus_fill"})
ROW_KEYS = (
    "id",
    "tier",
    "question",
    "answerable",
    "reference_answer",
    "gold_evidence",
    "probe",
    "status",
)
EVIDENCE_KEYS = ("doc_id", "locator", "note")
ID_RE = re.compile(r"^q\d{3}$")
# Matches both the doc_id form (TODO_ppo_paper_1) and the prose form
# ("TODO: fill after corpus assembly").
PLACEHOLDER_RE = re.compile(r"\bTODO", re.IGNORECASE)

ERROR = "error"
WARNING = "warning"


def has_placeholder(value: Any) -> bool:
    """Does this value (or anything inside it) still carry a ``TODO`` marker?"""
    return bool(PLACEHOLDER_RE.search(json.dumps(value, ensure_ascii=False)))


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    """One problem, addressed to a line so it can be opened and fixed."""

    severity: str
    line: int
    row_id: str
    field: str
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}"
        if self.row_id:
            where += f" [{self.row_id}]"
        if self.field:
            where += f" {self.field}"
        return f"{self.severity:>7}: {where}: {self.message}"


@dataclass
class ValidationReport:
    path: Path
    n_rows: int = 0
    findings: list[Finding] = field(default_factory=list)
    tiers: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    n_evidence: int = 0
    locators_checked: bool = False
    locator_resolutions: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_rows(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[Finding]]:
    """Parse JSONL into ``(line_number, row)`` pairs, reporting bad lines.

    A malformed line is reported and skipped rather than aborting, so one
    stray comma does not hide every other problem in the file.
    """
    rows: list[tuple[int, dict[str, Any]]] = []
    findings: list[Finding] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            findings.append(Finding(ERROR, line_no, "", "", f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(value, dict):
            findings.append(
                Finding(
                    ERROR, line_no, "", "", f"expected an object, got {type(value).__name__}"
                )
            )
            continue
        rows.append((line_no, value))
    return rows, findings


# --------------------------------------------------------------------------- #
# Locators
# --------------------------------------------------------------------------- #


def normalize_for_match(text: str) -> str:
    """Collapse whitespace runs, so a locator survives line wrapping.

    The parser joins wrapped PDF lines with single spaces, but an annotator
    copying a phrase out of a paper may carry a newline or a double space.
    Matching on whitespace-normalized text makes the locator robust to that
    without loosening it into fuzzy matching -- every other character still has
    to agree exactly.
    """
    return " ".join(text.split())


def locator_matches(locator: str, chunk_bodies: dict[str, str]) -> list[str]:
    """Chunk IDs whose body contains ``locator`` verbatim.

    Matches against the chunk *body*, not its text: the structure chunker
    prepends a heading breadcrumb that is not document content, and a locator
    is a phrase quoted from the document.
    """
    needle = normalize_for_match(locator)
    if not needle:
        return []
    return sorted(cid for cid, body in chunk_bodies.items() if needle in body)


# --------------------------------------------------------------------------- #
# Row validation
# --------------------------------------------------------------------------- #


def validate_row(
    line: int,
    row: dict[str, Any],
    *,
    known_doc_ids: frozenset[str],
    chunk_bodies: dict[str, str] | None = None,
) -> list[Finding]:
    """All checks for one row. Later checks are skipped when earlier ones fail,
    so a row with the wrong shape reports that once instead of cascading."""
    findings: list[Finding] = []
    row_id = row.get("id") if isinstance(row.get("id"), str) else ""

    def err(field_name: str, message: str) -> None:
        findings.append(Finding(ERROR, line, row_id, field_name, message))

    def warn(field_name: str, message: str) -> None:
        findings.append(Finding(WARNING, line, row_id, field_name, message))

    # --- keys -------------------------------------------------------------- #
    missing = [k for k in ROW_KEYS if k not in row]
    unknown = sorted(set(row) - set(ROW_KEYS))
    for key in missing:
        err(key, "missing required key")
    for key in unknown:
        err(key, f"unknown key (known keys: {list(ROW_KEYS)})")
    if missing:
        return findings

    # --- types and value domains ------------------------------------------- #
    if not isinstance(row["id"], str) or not ID_RE.match(row["id"]):
        err("id", f"expected a 'qNNN' string, got {row['id']!r}")
    for key in ("question", "probe"):
        if not isinstance(row[key], str):
            err(key, f"expected a string, got {type(row[key]).__name__}")
        elif not row[key].strip():
            err(key, "must not be empty")
    if not isinstance(row["answerable"], bool):
        err("answerable", f"expected a bool, got {type(row['answerable']).__name__}")
    if not isinstance(row["tier"], str) or row["tier"] not in TIERS:
        err("tier", f"{row['tier']!r} not in {sorted(TIERS)}")
    if not isinstance(row["status"], str) or row["status"] not in STATUSES:
        err("status", f"{row['status']!r} not in {sorted(STATUSES)}")
    if row["reference_answer"] is not None and not isinstance(row["reference_answer"], str):
        err(
            "reference_answer",
            f"expected a string or null, got {type(row['reference_answer']).__name__}",
        )
    if not isinstance(row["gold_evidence"], list):
        err(
            "gold_evidence",
            f"expected a list, got {type(row['gold_evidence']).__name__}",
        )
        return findings

    # --- evidence entries --------------------------------------------------- #
    for i, entry in enumerate(row["gold_evidence"]):
        where = f"gold_evidence[{i}]"
        if not isinstance(entry, dict):
            err(where, f"expected an object, got {type(entry).__name__}")
            continue
        for key in EVIDENCE_KEYS:
            if key not in entry:
                err(f"{where}.{key}", "missing required key")
        for key in sorted(set(entry) - set(EVIDENCE_KEYS)):
            err(f"{where}.{key}", f"unknown key (known keys: {list(EVIDENCE_KEYS)})")
        for key in EVIDENCE_KEYS:
            if key in entry and not isinstance(entry[key], str):
                err(f"{where}.{key}", f"expected a string, got {type(entry[key]).__name__}")
        # note may be empty (it is optional commentary); doc_id and locator may not.
        for key in ("doc_id", "locator"):
            if isinstance(entry.get(key), str) and not entry[key].strip():
                err(f"{where}.{key}", "must not be empty")

    # Duplicate (doc_id, locator) pairs double-count a single piece of evidence.
    seen: dict[tuple[str, str], int] = {}
    for i, entry in enumerate(row["gold_evidence"]):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("doc_id")), normalize_for_match(str(entry.get("locator"))))
        if key in seen:
            warn(
                f"gold_evidence[{i}]",
                f"duplicates gold_evidence[{seen[key]}] (same doc_id and locator)",
            )
        else:
            seen[key] = i

    # --- cross-field coherence ---------------------------------------------- #
    answerable = row["answerable"]
    evidence = row["gold_evidence"]
    reference = row["reference_answer"]
    # A wrongly-typed reference_answer was already reported above; comparing it
    # here as well would crash rather than add anything.
    reference_typed = reference is None or isinstance(reference, str)
    if isinstance(answerable, bool):
        if answerable and not evidence:
            err(
                "gold_evidence",
                "answerable: true but no gold evidence -- recall@k is undefined "
                "for this row",
            )
        if not answerable and evidence:
            err(
                "gold_evidence",
                f"answerable: false but {len(evidence)} gold evidence entr"
                f"{'y' if len(evidence) == 1 else 'ies'} -- an unanswerable "
                "question cannot have gold evidence",
            )
        if reference_typed and not answerable and reference is not None:
            err(
                "reference_answer",
                "answerable: false must have a null reference_answer "
                "(the expected behaviour is abstention)",
            )
        if reference_typed and answerable and not (reference or "").strip():
            err("reference_answer", "answerable: true needs a reference answer")
        if (row["tier"] == "unanswerable") != (answerable is False):
            err(
                "tier",
                f"tier {row['tier']!r} and answerable {answerable} disagree: "
                "tier 'unanswerable' and answerable false must go together",
            )

    # --- the TODO/status pairing -------------------------------------------- #
    placeholder_fields = sorted(
        key for key in ("reference_answer", "gold_evidence") if has_placeholder(row[key])
    )
    if row["status"] == "ready" and placeholder_fields:
        err(
            "status",
            f"status: ready but TODO placeholders remain in {placeholder_fields} -- "
            "a placeholder in a measured run silently corrupts the metric",
        )
    if row["status"] == "needs_corpus_fill" and not placeholder_fields:
        warn(
            "status",
            "status: needs_corpus_fill but no TODO placeholders remain; "
            "flip to 'ready' if the annotation is finished",
        )

    # --- corpus cross-reference --------------------------------------------- #
    for i, entry in enumerate(row["gold_evidence"]):
        if not isinstance(entry, dict):
            continue
        doc_id = entry.get("doc_id")
        locator = entry.get("locator")
        if not isinstance(doc_id, str) or not isinstance(locator, str):
            continue
        where = f"gold_evidence[{i}]"

        if has_placeholder(doc_id):
            # Unfilled by design; the status check above already covers it.
            continue
        if doc_id not in known_doc_ids:
            err(
                f"{where}.doc_id",
                f"{doc_id!r} is not in the corpus manifest -- gold evidence "
                "pointing outside the corpus can never be retrieved",
            )
            continue
        if chunk_bodies is None or has_placeholder(locator):
            continue

        hits = locator_matches(locator, chunk_bodies)
        same_doc = [cid for cid in hits if cid.rsplit("#", 1)[0] == doc_id]
        if not hits:
            err(
                f"{where}.locator",
                f"does not occur in any chunk of the active chunk set: "
                f"{locator[:60]!r} -- this row can never score above zero",
            )
        elif not same_doc:
            err(
                f"{where}.locator",
                f"occurs in the corpus but not in {doc_id} (found in "
                f"{', '.join(hits[:3])}) -- doc_id and locator disagree",
            )
        elif len(same_doc) > 4:
            warn(
                f"{where}.locator",
                f"resolves to {len(same_doc)} chunks in {doc_id}; a phrase this "
                "common makes recall@k easy to satisfy by accident",
            )

    return findings


# --------------------------------------------------------------------------- #
# File validation
# --------------------------------------------------------------------------- #


def validate(
    path: Path,
    *,
    known_doc_ids: frozenset[str],
    chunk_bodies: dict[str, str] | None = None,
) -> ValidationReport:
    """Validate the whole file. Never writes to ``path``."""
    report = ValidationReport(path=path)
    rows, parse_findings = read_rows(path)
    report.findings.extend(parse_findings)
    report.n_rows = len(rows)

    if not rows and not parse_findings:
        report.findings.append(Finding(ERROR, 0, "", "", "file contains no rows"))
        return report

    seen_ids: dict[str, int] = {}
    for line, row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                report.findings.append(
                    Finding(
                        ERROR,
                        line,
                        row_id,
                        "id",
                        f"duplicate id (first seen on line {seen_ids[row_id]})",
                    )
                )
            else:
                seen_ids[row_id] = line

        report.findings.extend(
            validate_row(
                line, row, known_doc_ids=known_doc_ids, chunk_bodies=chunk_bodies
            )
        )

        tier = row.get("tier")
        if isinstance(tier, str):
            report.tiers[tier] = report.tiers.get(tier, 0) + 1
        status = row.get("status")
        if isinstance(status, str):
            report.statuses[status] = report.statuses.get(status, 0) + 1
        evidence = row.get("gold_evidence")
        if isinstance(evidence, list):
            report.n_evidence += len(evidence)

    report.locators_checked = chunk_bodies is not None
    if chunk_bodies is not None:
        for _line, row in rows:
            for entry in row.get("gold_evidence") or []:
                if not isinstance(entry, dict):
                    continue
                locator = entry.get("locator")
                doc_id = entry.get("doc_id")
                if (
                    isinstance(locator, str)
                    and isinstance(doc_id, str)
                    and not has_placeholder(locator)
                    and not has_placeholder(doc_id)
                ):
                    report.locator_resolutions[f"{doc_id}: {locator[:40]}"] = len(
                        locator_matches(locator, chunk_bodies)
                    )
    return report
