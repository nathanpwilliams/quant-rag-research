import json
from pathlib import Path

import pytest

from quant_rag.ingest import download as dl

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/{aid}v{ver}</id>
    <published>2017-06-30T09:00:00Z</published>
    <title>{title}</title>
  </entry>
</feed>"""

EMPTY_FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


def _atom(aid, title, ver=3):
    return ATOM.format(aid=aid, ver=ver, title=title).encode()


def test_normalize_and_similarity():
    a = "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
    b = "A Deep Reinforcement Learning Framework for the\n  Financial Portfolio Management Problem."
    assert dl.title_similarity(a, b) > 0.98
    assert dl.title_similarity(a, "Playing Atari with Deep Reinforcement Learning") < 0.6


def test_pdf_url_pins_version():
    assert dl.pdf_url("1706.10059", 3) == "https://arxiv.org/pdf/1706.10059v3"
    assert dl.pdf_url("1706.10059") == "https://arxiv.org/pdf/1706.10059"


def test_parse_atom_extracts_version_and_title():
    meta = dl.parse_atom(_atom("1706.10059", "Some  Title\n Here", ver=2), "1706.10059")
    assert meta.version == 2
    assert meta.title == "Some Title Here"


def test_parse_atom_empty_feed_raises():
    with pytest.raises(dl.ArxivNotFound):
        dl.parse_atom(EMPTY_FEED.encode(), "0000.00000")


def test_parse_atom_wrong_id_raises():
    with pytest.raises(dl.ArxivNotFound):
        dl.parse_atom(_atom("1111.11111", "x"), "2222.22222")


@pytest.fixture
def workspace(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = raw / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_version": "0",
                "updated": "2026-01-01",
                "documents": [
                    {
                        "doc_id": "arxiv_1706.10059",
                        "title": "A Deep RL Framework for Portfolio Management",
                        "authors": "x",
                        "year": 2017,
                        "source_type": "arxiv",
                        "arxiv_id": "1706.10059",
                        "url": "u",
                        "tags": [],
                        "role": "r",
                        "status": "to_download",
                    },
                    {
                        "doc_id": "note_x",
                        "title": "n",
                        "authors": "x",
                        "year": 2026,
                        "source_type": "note",
                        "arxiv_id": None,
                        "url": None,
                        "tags": [],
                        "role": "r",
                        "status": "to_write",
                    },
                ],
            }
        )
    )
    return manifest, raw


def _fake_fetch(title, pdf=b"%PDF-1.4 fake"):
    calls = []

    def fetch(url):
        calls.append(url)
        if url.startswith(dl.ARXIV_API):
            return _atom("1706.10059", title)
        return pdf

    fetch.calls = calls
    return fetch


def test_download_writes_pdf_and_updates_manifest(workspace):
    manifest_path, raw = workspace
    fetch = _fake_fetch("A Deep RL Framework for Portfolio Management")
    report = dl.download_corpus(
        manifest_path=manifest_path, raw_dir=raw, delay=0, fetch=fetch, log=lambda s: None
    )
    assert [r.outcome for r in report.results] == ["downloaded"]
    assert fetch.calls[-1] == "https://arxiv.org/pdf/1706.10059v3"
    pdf = raw / "arxiv_1706.10059.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    doc = json.loads(manifest_path.read_text())["documents"][0]
    assert doc["status"] == "downloaded"
    assert doc["arxiv_version"] == 3
    assert doc["sha256"] == dl.sha256_file(pdf)
    assert "downloaded_at" in doc

    # second run: nothing to do, no network
    fetch2 = _fake_fetch("irrelevant")
    report2 = dl.download_corpus(
        manifest_path=manifest_path, raw_dir=raw, delay=0, fetch=fetch2, log=lambda s: None
    )
    assert report2.results == [] and fetch2.calls == []


def test_title_mismatch_blocks_download(workspace):
    manifest_path, raw = workspace
    fetch = _fake_fetch("Playing Atari with Deep Reinforcement Learning")
    report = dl.download_corpus(
        manifest_path=manifest_path, raw_dir=raw, delay=0, fetch=fetch, log=lambda s: None
    )
    assert report.results[0].outcome == "title_mismatch"
    assert report.failed
    assert not (raw / "arxiv_1706.10059.pdf").exists()
    assert json.loads(manifest_path.read_text())["documents"][0]["status"] == "to_download"


def test_non_pdf_payload_is_rejected(workspace):
    manifest_path, raw = workspace
    fetch = _fake_fetch("A Deep RL Framework for Portfolio Management", pdf=b"<html>no</html>")
    report = dl.download_corpus(
        manifest_path=manifest_path, raw_dir=raw, delay=0, fetch=fetch, log=lambda s: None
    )
    assert report.results[0].outcome == "error"
    assert not (raw / "arxiv_1706.10059.pdf").exists()


def test_dry_run_touches_nothing(workspace):
    manifest_path, raw = workspace
    before = manifest_path.read_text()
    fetch = _fake_fetch("A Deep RL Framework for Portfolio Management")
    report = dl.download_corpus(
        manifest_path=manifest_path, raw_dir=raw, delay=0, dry_run=True, fetch=fetch,
        log=lambda s: None,
    )
    assert report.results[0].outcome == "skipped"
    assert len(fetch.calls) == 1  # metadata only
    assert manifest_path.read_text() == before
