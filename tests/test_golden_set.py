"""Golden-set validator tests.

Every test here builds its own throwaway JSONL in ``tmp_path``. The real
``eval/golden_set*.jsonl`` is human-annotated (CLAUDE.md rule 1) and is only
ever *read* -- by the last test in this file, which asserts the committed file
passes.

The point of a validator is the errors it catches, so each rule gets a row that
breaks exactly that rule and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_rag.golden_set import (
    ERROR,
    WARNING,
    Finding,
    has_placeholder,
    locator_matches,
    normalize_for_match,
    validate,
)

DOC_IDS = frozenset({"arxiv_1706.10059", "note_afml_labeling"})


def row(**overrides) -> dict:
    """A valid, finished row; override one field to make it invalid."""
    base = {
        "id": "q001",
        "tier": "lookup",
        "question": "What network architectures instantiate EIIE?",
        "answerable": True,
        "reference_answer": "A CNN, a basic RNN, and an LSTM.",
        "gold_evidence": [
            {
                "doc_id": "arxiv_1706.10059",
                "locator": "Ensemble of Identical Independent Evaluators",
                "note": "architecture section",
            }
        ],
        "probe": "single-hop factual lookup",
        "status": "ready",
    }
    base.update(overrides)
    return base


def unanswerable_row(**overrides) -> dict:
    base = {
        "id": "q010",
        "tier": "unanswerable",
        "question": "What Sharpe does the Medallion Fund achieve per this corpus?",
        "answerable": False,
        "reference_answer": None,
        "gold_evidence": [],
        "probe": "abstention test",
        "status": "ready",
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, *rows, name: str = "golden.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def check(tmp_path: Path, *rows, chunk_bodies=None):
    return validate(
        write(tmp_path, *rows), known_doc_ids=DOC_IDS, chunk_bodies=chunk_bodies
    )


def messages(report, severity=ERROR) -> str:
    return " | ".join(
        f"{f.field}: {f.message}" for f in report.findings if f.severity == severity
    )


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_valid_file_passes_clean(tmp_path):
    report = check(tmp_path, row(), unanswerable_row())
    assert report.ok, messages(report)
    assert report.findings == []
    assert report.n_rows == 2
    assert report.n_evidence == 1
    assert report.tiers == {"lookup": 1, "unanswerable": 1}
    assert report.statuses == {"ready": 2}


def test_an_unfilled_row_with_matching_status_passes(tmp_path):
    """A TODO row is not an error -- it is work in progress, correctly flagged."""
    report = check(
        tmp_path,
        row(
            id="q005",
            reference_answer="TODO: fill after corpus assembly",
            gold_evidence=[
                {"doc_id": "TODO_ppo_paper_1", "locator": "TODO: reward phrase", "note": ""}
            ],
            status="needs_corpus_fill",
        ),
    )
    assert report.ok, messages(report)
    assert report.statuses == {"needs_corpus_fill": 1}


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_malformed_json_is_reported_with_a_line_number_and_does_not_abort(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(row()) + "\n" + "{not json,}\n" + json.dumps(unanswerable_row()) + "\n",
        encoding="utf-8",
    )
    report = validate(path, known_doc_ids=DOC_IDS)
    assert not report.ok
    assert [f.line for f in report.errors] == [2]
    assert "invalid JSON" in report.errors[0].message
    # The two good rows were still validated rather than lost.
    assert report.n_rows == 2


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(f"\n{json.dumps(row())}\n\n", encoding="utf-8")
    report = validate(path, known_doc_ids=DOC_IDS)
    assert report.ok, messages(report)
    assert report.n_rows == 1


def test_a_non_object_line_is_an_error(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text('["not", "an object"]\n', encoding="utf-8")
    report = validate(path, known_doc_ids=DOC_IDS)
    assert not report.ok
    assert "expected an object" in report.errors[0].message


def test_an_empty_file_is_an_error(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("", encoding="utf-8")
    report = validate(path, known_doc_ids=DOC_IDS)
    assert not report.ok
    assert "no rows" in report.errors[0].message


def test_missing_and_unknown_keys_are_both_errors(tmp_path):
    bad = row()
    del bad["probe"]
    bad["gold_evidenc"] = []  # the typo this check exists for
    report = check(tmp_path, bad)
    assert not report.ok
    assert "probe: missing required key" in messages(report)
    assert "gold_evidenc: unknown key" in messages(report)


def test_duplicate_ids_are_an_error(tmp_path):
    report = check(tmp_path, row(), row())
    assert not report.ok
    assert "duplicate id" in messages(report)
    assert "first seen on line 1" in messages(report)


@pytest.mark.parametrize("bad_id", ["Q001", "q1", "question-1", "q0001", ""])
def test_ids_must_look_like_qNNN(tmp_path, bad_id):
    report = check(tmp_path, row(id=bad_id))
    assert not report.ok
    assert "expected a 'qNNN' string" in messages(report)


def test_unknown_tier_and_status_are_errors(tmp_path):
    assert "tier" in messages(check(tmp_path, row(tier="multi_hop")))
    assert "status" in messages(check(tmp_path, row(status="done")))


def test_wrong_types_are_errors(tmp_path):
    assert "expected a bool" in messages(check(tmp_path, row(answerable="yes")))
    assert "expected a list" in messages(check(tmp_path, row(gold_evidence={})))
    assert "expected a string" in messages(check(tmp_path, row(question=42)))
    assert "expected a string or null" in messages(check(tmp_path, row(reference_answer=1)))


def test_empty_question_or_probe_is_an_error(tmp_path):
    assert "must not be empty" in messages(check(tmp_path, row(question="   ")))
    assert "must not be empty" in messages(check(tmp_path, row(probe="")))


def test_evidence_entry_shape_is_checked(tmp_path):
    report = check(
        tmp_path,
        row(gold_evidence=[{"doc_id": "arxiv_1706.10059", "note": "x", "page": 3}]),
    )
    assert not report.ok
    assert "gold_evidence[0].locator: missing required key" in messages(report)
    assert "gold_evidence[0].page: unknown key" in messages(report)


def test_empty_doc_id_or_locator_is_an_error_but_empty_note_is_fine(tmp_path):
    report = check(
        tmp_path, row(gold_evidence=[{"doc_id": "arxiv_1706.10059", "locator": " ", "note": ""}])
    )
    assert "gold_evidence[0].locator: must not be empty" in messages(report)
    assert "note" not in messages(report)


def test_a_non_object_evidence_entry_is_an_error(tmp_path):
    report = check(tmp_path, row(gold_evidence=["arxiv_1706.10059"]))
    assert not report.ok
    assert "expected an object" in messages(report)


# --------------------------------------------------------------------------- #
# Cross-field coherence: the checks that protect the metric
# --------------------------------------------------------------------------- #


def test_answerable_true_without_gold_evidence_is_an_error(tmp_path):
    report = check(tmp_path, row(gold_evidence=[]))
    assert not report.ok
    assert "recall@k is undefined" in messages(report)


def test_answerable_false_with_gold_evidence_is_an_error(tmp_path):
    report = check(
        tmp_path,
        unanswerable_row(
            gold_evidence=[
                {"doc_id": "arxiv_1706.10059", "locator": "price tensor", "note": ""}
            ]
        ),
    )
    assert not report.ok
    assert "cannot have gold evidence" in messages(report)


def test_answerable_false_must_have_a_null_reference_answer(tmp_path):
    report = check(tmp_path, unanswerable_row(reference_answer="Actually it's 39%."))
    assert not report.ok
    assert "null reference_answer" in messages(report)


def test_answerable_true_needs_a_reference_answer(tmp_path):
    assert "needs a reference answer" in messages(check(tmp_path, row(reference_answer=None)))
    assert "needs a reference answer" in messages(check(tmp_path, row(reference_answer="  ")))


def test_tier_and_answerable_must_agree(tmp_path):
    """'unanswerable' is the tier the abstention metric reads; a mismatch would
    score an answerable row as an abstention test or vice versa."""
    report = check(tmp_path, row(tier="unanswerable"))
    assert not report.ok
    assert "disagree" in messages(report)

    report = check(tmp_path, unanswerable_row(tier="lookup"))
    assert not report.ok
    assert "disagree" in messages(report)


def test_duplicate_evidence_entries_warn(tmp_path):
    entry = {
        "doc_id": "arxiv_1706.10059",
        "locator": "Ensemble of Identical Independent Evaluators",
        "note": "",
    }
    report = check(tmp_path, row(gold_evidence=[entry, dict(entry, note="again")]))
    assert report.ok  # not fatal, but double-counts one piece of evidence
    assert "duplicates gold_evidence[0]" in messages(report, WARNING)


# --------------------------------------------------------------------------- #
# The TODO/status pairing
# --------------------------------------------------------------------------- #


def test_ready_with_a_todo_placeholder_is_an_error(tmp_path):
    """The costly mistake: a placeholder entering a measured run."""
    report = check(tmp_path, row(reference_answer="TODO: write this up"))
    assert not report.ok
    assert "status: ready but TODO placeholders remain" in messages(report)
    assert "reference_answer" in messages(report)


def test_ready_with_a_todo_doc_id_is_an_error(tmp_path):
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {"doc_id": "TODO_ppo_paper_1", "locator": "reward definition", "note": ""}
            ]
        ),
    )
    assert not report.ok
    assert "gold_evidence" in messages(report)


def test_needs_corpus_fill_without_any_todo_only_warns(tmp_path):
    report = check(tmp_path, row(status="needs_corpus_fill"))
    assert report.ok
    assert "flip to 'ready'" in messages(report, WARNING)


def test_has_placeholder_recognizes_both_conventions():
    assert has_placeholder("TODO_ppo_paper_1")
    assert has_placeholder("TODO: fill after corpus assembly")
    assert has_placeholder([{"doc_id": "TODO_x"}])
    assert has_placeholder("todo: lowercase still counts")
    assert not has_placeholder("Ensemble of Identical Independent Evaluators")
    assert not has_placeholder(None)
    # "todos" as a word should not trip it; the marker is the token TODO.
    assert has_placeholder("TODOs remain")


# --------------------------------------------------------------------------- #
# Corpus cross-reference
# --------------------------------------------------------------------------- #


def test_a_doc_id_outside_the_manifest_is_an_error(tmp_path):
    """The alphapulse-removal scenario: a doc leaves the corpus and the
    annotation pointing at it becomes permanently unretrievable."""
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {"doc_id": "note_alphapulse_mdp", "locator": "state design", "note": ""}
            ]
        ),
    )
    assert not report.ok
    assert "not in the corpus manifest" in messages(report)


def test_todo_doc_ids_are_not_checked_against_the_manifest(tmp_path):
    report = check(
        tmp_path,
        row(
            reference_answer="TODO: later",
            gold_evidence=[
                {"doc_id": "TODO_ppo_paper_1", "locator": "TODO: phrase", "note": ""}
            ],
            status="needs_corpus_fill",
        ),
    )
    assert report.ok, messages(report)


# --------------------------------------------------------------------------- #
# Locator resolution
# --------------------------------------------------------------------------- #

CHUNKS = {
    "arxiv_1706.10059#0000": normalize_for_match(
        "we propose an Ensemble of Identical Independent Evaluators topology"
    ),
    "arxiv_1706.10059#0001": normalize_for_match("the price tensor is normalized"),
    "note_afml_labeling#0000": normalize_for_match("triple-barrier labeling sets barriers"),
}


def test_a_locator_that_resolves_is_clean(tmp_path):
    report = check(tmp_path, row(), chunk_bodies=CHUNKS)
    assert report.ok, messages(report)
    assert report.locators_checked
    assert list(report.locator_resolutions.values()) == [1]


def test_a_locator_that_matches_nothing_is_an_error(tmp_path):
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {
                    "doc_id": "arxiv_1706.10059",
                    "locator": "a phrase the parser never produced",
                    "note": "",
                }
            ]
        ),
        chunk_bodies=CHUNKS,
    )
    assert not report.ok
    assert "does not occur in any chunk" in messages(report)
    assert "can never score above zero" in messages(report)


def test_a_locator_in_the_wrong_document_is_an_error(tmp_path):
    """The locator is real but attributed to the wrong doc_id -- a transcription
    slip that would score as a retrieval miss forever."""
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {
                    "doc_id": "note_afml_labeling",
                    "locator": "the price tensor is normalized",
                    "note": "",
                }
            ]
        ),
        chunk_bodies=CHUNKS,
    )
    assert not report.ok
    assert "occurs in the corpus but not in note_afml_labeling" in messages(report)


def test_an_over_common_locator_warns(tmp_path):
    bodies = {f"arxiv_1706.10059#{i:04d}": "the price tensor appears here" for i in range(6)}
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {"doc_id": "arxiv_1706.10059", "locator": "price tensor", "note": ""}
            ]
        ),
        chunk_bodies=bodies,
    )
    assert report.ok  # usable, just weak evidence
    assert "resolves to 6 chunks" in messages(report, WARNING)


def test_locator_matching_survives_line_wrapping():
    """An annotator's copied phrase may carry a newline the parser does not."""
    bodies = {"d#0000": normalize_for_match("the reward is a volatility scaled return")}
    assert locator_matches("volatility\n  scaled   return", bodies) == ["d#0000"]
    assert locator_matches("volatility scaled return", bodies) == ["d#0000"]
    # Normalization is whitespace-only; it is not fuzzy matching.
    assert locator_matches("volatility-scaled return", bodies) == []
    assert locator_matches("", bodies) == []


def test_normalize_for_match_collapses_whitespace_only():
    assert normalize_for_match("a\n\nb   c\t d") == "a b c d"
    assert normalize_for_match("  padded  ") == "padded"
    assert normalize_for_match("Case Sensitive") == "Case Sensitive"


def test_locators_are_not_checked_when_no_chunk_set_is_supplied(tmp_path):
    report = check(
        tmp_path,
        row(
            gold_evidence=[
                {"doc_id": "arxiv_1706.10059", "locator": "nonexistent phrase", "note": ""}
            ]
        ),
    )
    assert report.ok, messages(report)
    assert not report.locators_checked


# --------------------------------------------------------------------------- #
# Findings are addressable
# --------------------------------------------------------------------------- #


def test_findings_name_the_line_the_row_and_the_field(tmp_path):
    report = check(tmp_path, row(), row(id="q002", gold_evidence=[]))
    finding = report.errors[0]
    assert finding.line == 2
    assert finding.row_id == "q002"
    assert finding.field == "gold_evidence"
    rendered = str(finding)
    assert "line 2" in rendered and "q002" in rendered and "gold_evidence" in rendered


def test_finding_renders_without_a_row_id(tmp_path):
    assert "line 7" in str(Finding(ERROR, 7, "", "", "invalid JSON"))


# --------------------------------------------------------------------------- #
# The committed file
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_golden_set_is_schema_valid():
    """Read-only (CLAUDE.md rule 1): this asserts, it never rewrites.

    Locators are checked separately, since the chunk set is gitignored.
    """
    from quant_rag.ingest.manifest import iter_documents, load_manifest

    path = REPO_ROOT / "eval" / "golden_set_seed.jsonl"
    if not path.exists():  # pragma: no cover - the file is committed
        pytest.skip(f"{path} not found")

    doc_ids = frozenset(d["doc_id"] for d in iter_documents(load_manifest()))
    report = validate(path, known_doc_ids=doc_ids)
    assert report.ok, messages(report)
    assert report.n_rows >= 10


def test_the_committed_golden_sets_locators_resolve_under_the_active_config():
    """Every filled-in locator must resolve, or that row can never score.

    Skipped when the corpus has not been ingested, since chunks are gitignored.
    """
    from quant_rag.config import load_config
    from quant_rag.index.store import chunks_path, load_chunks
    from quant_rag.ingest.manifest import iter_documents, load_manifest

    path = REPO_ROOT / "eval" / "golden_set_seed.jsonl"
    config = load_config(REPO_ROOT / "config" / "pipeline.yaml")
    try:
        chunks = load_chunks(chunks_path(config))
    except Exception:  # noqa: BLE001 - not ingested yet
        pytest.skip("no chunk set on disk; run `uv run quant-rag ingest` first")

    bodies = {c.chunk_id: normalize_for_match(c.body) for c in chunks}
    doc_ids = frozenset(d["doc_id"] for d in iter_documents(load_manifest()))
    report = validate(path, known_doc_ids=doc_ids, chunk_bodies=bodies)
    assert report.ok, messages(report)
    assert report.locator_resolutions
    assert all(n > 0 for n in report.locator_resolutions.values()), (
        f"unresolved locators: "
        f"{[k for k, n in report.locator_resolutions.items() if not n]}"
    )
