import copy

from quant_rag.ingest.manifest import (
    DEFAULT_MANIFEST_PATH,
    iter_documents,
    load_manifest,
    validate_manifest,
)


def test_committed_manifest_is_valid():
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    assert validate_manifest(manifest) == []
    assert len(manifest["documents"]) > 0


def test_iter_documents_filters():
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    arxiv = list(iter_documents(manifest, source_type="arxiv"))
    assert arxiv and all(d["source_type"] == "arxiv" for d in arxiv)
    notes = list(iter_documents(manifest, source_type="note", status="to_write"))
    assert all(d["doc_id"].startswith("note_") for d in notes)


def _one(doc_id="arxiv_1706.10059", **overrides):
    doc = {
        "doc_id": doc_id,
        "title": "t",
        "authors": "a",
        "year": 2017,
        "source_type": "arxiv",
        "arxiv_id": "1706.10059",
        "url": "https://arxiv.org/abs/1706.10059",
        "tags": [],
        "role": "r",
        "status": "to_download",
    }
    doc.update(overrides)
    return doc


def _manifest(*docs):
    return {"corpus_version": "0", "updated": "2026-01-01", "documents": list(docs)}


def test_validate_catches_duplicate_ids():
    errors = validate_manifest(_manifest(_one(), copy.deepcopy(_one())))
    assert any("duplicate" in e for e in errors)


def test_validate_catches_arxiv_id_mismatch():
    errors = validate_manifest(_manifest(_one(arxiv_id="9999.99999")))
    assert any("doc_id must equal" in e for e in errors)


def test_validate_catches_bad_status_and_prefix():
    errors = validate_manifest(_manifest(_one(doc_id="web_x", status="done")))
    assert any("unknown status" in e for e in errors)
    assert any("prefix does not match" in e for e in errors)


def test_validate_requires_null_arxiv_id_for_notes():
    doc = _one(doc_id="note_x", source_type="note", arxiv_id="1", status="to_write")
    errors = validate_manifest(_manifest(doc))
    assert any("arxiv_id null" in e for e in errors)
