import pytest

from quant_rag.config import ChunkingConfig
from quant_rag.ingest.chunk import (
    FixedChunker,
    StructureChunker,
    build_chunker,
    chunk_id,
    fit_to_budget,
    window_indices,
)
from quant_rag.ingest.manifest import DEFAULT_RAW_DIR, iter_documents, load_manifest
from quant_rag.ingest.parse import Element, ParsedDoc, parse_document, render
from quant_rag.ingest.tokenizer import RegexTokenizer, get_tokenizer

TOKENIZER = RegexTokenizer()


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_doc(elements: list[Element], doc_id: str = "note_test") -> ParsedDoc:
    text, sections = render(elements)
    return ParsedDoc(
        doc_id=doc_id,
        title="t",
        source_type="note",
        source_file="x.md",
        parser="markdown",
        text=text,
        sections=sections,
    )


def tokens_of(text: str) -> list[str]:
    return [text[a:b] for a, b in TOKENIZER.spans(text)]


# --------------------------------------------------------------------------- #
# window_indices: the boundary logic both chunkers share
# --------------------------------------------------------------------------- #


def test_window_indices_covers_every_token():
    windows = window_indices(250, 100, 20)
    assert windows[0] == (0, 100)
    assert windows[-1][1] == 250
    covered = {i for start, end in windows for i in range(start, end)}
    assert covered == set(range(250))


def test_window_indices_short_document_is_one_window():
    assert window_indices(30, 100, 20) == [(0, 30)]
    assert window_indices(0, 100, 20) == []


def test_window_indices_exact_multiple_does_not_emit_a_trailing_duplicate():
    """Stopping at the end is what keeps the last window from repeating."""
    windows = window_indices(100, 100, 20)
    assert windows == [(0, 100)]
    windows = window_indices(180, 100, 20)
    assert windows == [(0, 100), (80, 180)]


def test_window_indices_overlap_is_exact():
    windows = window_indices(500, 100, 20)
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[1] - later[0] == 20


def test_window_indices_without_overlap_is_a_partition():
    windows = window_indices(250, 100, 0)
    assert windows == [(0, 100), (100, 200), (200, 250)]


# --------------------------------------------------------------------------- #
# fixed
# --------------------------------------------------------------------------- #


@pytest.fixture
def fixed() -> FixedChunker:
    return FixedChunker(chunk_tokens=50, overlap_tokens=10, tokenizer=TOKENIZER)


def test_fixed_respects_the_token_budget(fixed):
    doc = make_doc([Element(words(400), 0)])
    chunks = fixed.chunk(doc)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 50 for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)


def test_fixed_overlap_carries_the_tail_of_the_previous_chunk(fixed):
    doc = make_doc([Element(words(300), 0)])
    chunks = fixed.chunk(doc)
    for earlier, later in zip(chunks, chunks[1:]):
        assert tokens_of(earlier.text)[-10:] == tokens_of(later.text)[:10]


def test_fixed_chunks_are_exact_slices_of_the_document(fixed):
    doc = make_doc([Element("1 Method", 1), Element(words(300), 0)])
    for chunk in fixed.chunk(doc):
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.prefix == ""
        assert not chunk.text[0].isspace() and not chunk.text[-1].isspace()


def test_fixed_covers_the_whole_document(fixed):
    doc = make_doc([Element(words(300), 0)])
    chunks = fixed.chunk(doc)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)


def test_fixed_attributes_chunks_to_the_section_they_start_in(fixed):
    doc = make_doc(
        [
            Element("1 Method", 1),
            Element(words(120, "m"), 0),
            Element("2 Results", 1),
            Element(words(120, "r"), 0),
        ]
    )
    chunks = fixed.chunk(doc)
    assert chunks[0].section_title == "1 Method"
    assert chunks[0].section_path == ("1 Method",)
    assert any(c.section_title == "2 Results" for c in chunks)


def test_chunk_ids_are_deterministic_and_positional(fixed):
    doc = make_doc([Element(words(300), 0)], doc_id="arxiv_1706.10059")
    first = fixed.chunk(doc)
    second = fixed.chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "arxiv_1706.10059#0000"
    assert [c.chunk_id for c in first] == [
        chunk_id("arxiv_1706.10059", i) for i in range(len(first))
    ]
    assert [c.index for c in first] == list(range(len(first)))


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


@pytest.fixture
def structure() -> StructureChunker:
    return StructureChunker(chunk_tokens=50, overlap_tokens=10, tokenizer=TOKENIZER)


def test_structure_prepends_the_section_breadcrumb(structure):
    doc = make_doc(
        [
            Element("3 Method", 1),
            Element("3.1 Reward design", 2),
            Element(words(30, "r"), 0),
        ]
    )
    chunk = structure.chunk(doc)[0]
    assert chunk.prefix == "3 Method > 3.1 Reward design\n\n"
    assert chunk.text.startswith("3 Method > 3.1 Reward design\n\n")
    assert chunk.section_title == "3.1 Reward design"
    assert chunk.section_path == ("3 Method", "3.1 Reward design")


def test_structure_body_excludes_the_heading_line_and_is_an_exact_slice(structure):
    doc = make_doc([Element("3 Method", 1), Element(words(30, "m"), 0)])
    chunk = structure.chunk(doc)[0]
    assert doc.text[chunk.char_start : chunk.char_end] == chunk.body
    assert not chunk.body.startswith("#")
    assert chunk.body.startswith("m0")


def test_structure_splits_on_section_boundaries(structure):
    doc = make_doc(
        [
            Element("1 Alpha", 1),
            Element(words(40, "a"), 0),
            Element("2 Beta", 1),
            Element(words(40, "b"), 0),
        ]
    )
    chunks = structure.chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].section_title == "1 Alpha"
    assert chunks[1].section_title == "2 Beta"
    assert "b0" not in chunks[0].body and "a0" not in chunks[1].body


def test_structure_packs_small_sections_but_not_across_top_level_sections(structure):
    doc = make_doc(
        [
            Element("1 Alpha", 1),
            Element("1.1 One", 2),
            Element(words(8, "a"), 0),
            Element("1.2 Two", 2),
            Element(words(8, "b"), 0),
            Element("2 Beta", 1),
            Element(words(8, "c"), 0),
        ]
    )
    chunks = structure.chunk(doc)
    # The two subsections of "1 Alpha" are packed together...
    assert any("a0" in c.body and "b0" in c.body for c in chunks)
    # ...but nothing mixes "1 Alpha" content with "2 Beta" content.
    assert not any("a0" in c.body and "c0" in c.body for c in chunks)


def test_structure_packed_chunk_uses_the_common_ancestor_as_prefix(structure):
    doc = make_doc(
        [
            Element("1 Alpha", 1),
            Element("1.1 One", 2),
            Element(words(8, "a"), 0),
            Element("1.2 Two", 2),
            Element(words(8, "b"), 0),
        ]
    )
    packed = next(c for c in structure.chunk(doc) if "a0" in c.body and "b0" in c.body)
    assert packed.prefix == "1 Alpha\n\n"


def test_structure_windows_an_oversized_section_with_overlap(structure):
    doc = make_doc([Element("1 Long", 1), Element(words(300, "x"), 0)])
    chunks = structure.chunk(doc)
    assert len(chunks) > 1
    assert all(c.prefix == "1 Long\n\n" for c in chunks)
    assert all(c.n_tokens <= 50 for c in chunks)
    # Overlap is measured on the body, since each part repeats the prefix.
    for earlier, later in zip(chunks, chunks[1:]):
        assert tokens_of(earlier.body)[-10:] == tokens_of(later.body)[:10]


def test_structure_chunks_are_exact_slices_and_ids_are_sequential(structure):
    doc = make_doc(
        [
            Element("1 Alpha", 1),
            Element(words(200, "a"), 0),
            Element("2 Beta", 1),
            Element(words(30, "b"), 0),
        ],
        doc_id="arxiv_1808.03668",
    )
    chunks = structure.chunk(doc)
    assert [c.chunk_id for c in chunks] == [
        chunk_id("arxiv_1808.03668", i) for i in range(len(chunks))
    ]
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.body


def test_structure_handles_a_document_with_no_headings(structure):
    doc = make_doc([Element(words(120), 0)])
    chunks = structure.chunk(doc)
    assert chunks
    assert all(c.prefix == "" for c in chunks)
    assert all(c.n_tokens <= 50 for c in chunks)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("strategy", "expected"), [("fixed", FixedChunker), ("structure", StructureChunker)]
)
def test_build_chunker_follows_the_config(strategy, expected):
    cfg = ChunkingConfig(
        strategy=strategy, chunk_tokens=128, overlap_tokens=16, tokenizer="regex"
    )
    chunker = build_chunker(cfg, TOKENIZER)
    assert isinstance(chunker, expected)
    assert (chunker.chunk_tokens, chunker.overlap_tokens) == (128, 16)


# --------------------------------------------------------------------------- #
# A real paper (skipped when the gitignored corpus is absent)
# --------------------------------------------------------------------------- #

REAL_DOC_ID = "arxiv_1706.10059"
LOCATOR = "Ensemble of Identical Independent Evaluators"


@pytest.fixture
def real_paper():
    doc = next(
        (d for d in iter_documents(load_manifest()) if d["doc_id"] == REAL_DOC_ID), None
    )
    if doc is None:
        pytest.skip(f"{REAL_DOC_ID} not in manifest")
    if not (DEFAULT_RAW_DIR / f"{REAL_DOC_ID}.pdf").exists():
        pytest.skip("corpus PDFs are gitignored; run `uv run quant-rag download` first")
    return parse_document(doc, DEFAULT_RAW_DIR)


@pytest.mark.parametrize("strategy", ["fixed", "structure"])
def test_real_paper_chunks_support_locator_resolution(real_paper, strategy):
    cfg = ChunkingConfig(
        strategy=strategy,
        chunk_tokens=512,
        overlap_tokens=64,
        tokenizer="BAAI/bge-base-en-v1.5",
    )
    chunks = build_chunker(cfg, TOKENIZER).chunk(real_paper)

    assert len(chunks) > 10
    assert all(c.doc_id == REAL_DOC_ID for c in chunks)
    assert all(c.strategy == strategy for c in chunks)
    assert all(c.n_tokens <= 512 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    # Every chunk body is a verbatim slice, which is what a locator matches.
    for chunk in chunks:
        assert real_paper.text[chunk.char_start : chunk.char_end] == chunk.body

    # Chunks run in document order.
    assert [c.char_start for c in chunks] == sorted(c.char_start for c in chunks)

    # The golden-set locator resolves to at least one chunk.
    hits = [c for c in chunks if LOCATOR in c.text]
    assert hits
    assert all(real_paper.text[h.char_start : h.char_end] == h.body for h in hits)

    if strategy == "structure":
        assert any(c.prefix for c in chunks)
        assert any("Abstract" in c.prefix for c in chunks)


# --------------------------------------------------------------------------- #
# fit_to_budget: n_tokens must describe the slice, not the document
# --------------------------------------------------------------------------- #


def test_fit_to_budget_is_a_noop_when_the_slice_already_fits():
    text = words(20)
    lo, hi, n = fit_to_budget(TOKENIZER, text, 0, len(text), 100)
    assert (lo, hi) == (0, len(text))
    assert n == TOKENIZER.count(text)


def test_fit_to_budget_shrinks_an_oversized_slice_and_reports_the_real_count():
    text = words(200)
    lo, hi, n = fit_to_budget(TOKENIZER, text, 0, len(text), 50)
    assert n <= 50
    assert hi < len(text)
    assert TOKENIZER.count(text[lo:hi]) == n
    assert not text[hi - 1].isspace()


def test_fit_to_budget_respects_a_nonzero_start_offset():
    text = words(200)
    start = text.index("w100")
    lo, hi, n = fit_to_budget(TOKENIZER, text, start, len(text), 30)
    assert lo == start
    assert n <= 30
    assert TOKENIZER.count(text[lo:hi]) == n


@pytest.mark.parametrize("strategy", ["fixed", "structure"])
def test_recorded_n_tokens_matches_a_recount_of_the_emitted_text(real_paper, strategy):
    """The bug this guards: windowing counted tokens in document context, so a
    chunk opening mid-word ("statistic" sliced to "istic") re-tokenized one
    token longer than recorded and the encoder silently dropped its tail.

    Uses the real BGE tokenizer, since WordPiece subword splitting is the whole
    mechanism -- the regex fallback has no subwords and cannot show the drift.
    """
    tokenizer = get_tokenizer("BAAI/bge-base-en-v1.5")
    if tokenizer.approximate:
        pytest.skip("exact BGE tokenizer unavailable; drift needs WordPiece")

    cfg = ChunkingConfig(
        strategy=strategy,
        chunk_tokens=510,
        overlap_tokens=64,
        tokenizer="BAAI/bge-base-en-v1.5",
    )
    chunks = build_chunker(cfg, tokenizer).chunk(real_paper)
    assert chunks

    for chunk in chunks:
        recount = tokenizer.count(chunk.text)
        assert recount == chunk.n_tokens, (
            f"{chunk.chunk_id}: recorded {chunk.n_tokens} tokens, text "
            f"recounts to {recount}"
        )
        # 510 content tokens + [CLS] + [SEP] is exactly the model's 512 window.
        assert recount <= 510, f"{chunk.chunk_id} would be truncated by the encoder"
