import json

import pytest

from quant_rag.ingest.manifest import DEFAULT_RAW_DIR, iter_documents, load_manifest
from quant_rag.ingest.parse import (
    Block,
    Element,
    Line,
    body_font_size,
    font_is_bold,
    font_is_italic,
    heading_level,
    is_drop_cap,
    is_margin_stamp,
    join_wrapped_lines,
    merge_split_headings,
    normalize_text,
    order_blocks,
    parse_document,
    parse_markdown,
    quality_warnings,
    render,
)

BODY = 10.0


def line(text, size=BODY, bold=False, italic=False, x0=0.0, x1=100.0, y0=0.0):
    return Line(text=text, size=size, bold=bold, italic=italic, x0=x0, x1=x1, y0=y0)


def block(*texts, x0=0.0, x1=100.0, y0=0.0, y1=10.0, size=BODY):
    return Block(
        lines=[line(t, size=size, x0=x0, x1=x1, y0=y0) for t in texts],
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #


def test_normalize_text_expands_ligatures_and_strips_controls():
    assert normalize_text("efﬁcient ﬂow") == "efficient flow"
    assert normalize_text("a\x07b") == "ab"
    assert normalize_text("a\r\nb") == "a\nb"
    assert normalize_text("soft­hyphen") == "softhyphen"


def test_join_wrapped_lines_undoes_hyphenation_but_keeps_compounds():
    assert join_wrapped_lines(["regu-", "larization term"]) == "regularization term"
    assert join_wrapped_lines(["risk-", "Adjusted return"]) == "risk- Adjusted return"
    assert join_wrapped_lines(["one", "", "two"]) == "one two"


# --------------------------------------------------------------------------- #
# Font signals
# --------------------------------------------------------------------------- #


def test_bold_and_italic_fall_back_to_font_names():
    """TeX Type1 fonts often omit the descriptor bits PyMuPDF reads."""
    assert font_is_bold("CMBX10", 4) is True
    assert font_is_bold("NimbusRomNo9L-Regu", 4) is False
    assert font_is_bold("Helvetica", 1 << 4) is True
    assert font_is_italic("NimbusRomNo9L-ReguItal", 4) is True
    assert font_is_italic("Times-Oblique", 0) is True
    assert font_is_italic("NimbusRomNo9L-Regu", 4) is False


def test_body_font_size_is_the_commonest_weighted_by_characters():
    pages = [[block("x" * 500), block("T", size=24.0)]]
    assert body_font_size(pages) == BODY


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #


def test_heading_level_from_numbering_when_bold():
    # Level comes from the numbering depth, not the font size.
    assert heading_level(line("2.1 Trading Period", bold=True), BODY) == 2
    assert heading_level(line("3.2.1 Reward", bold=True), BODY) == 3
    assert heading_level(line("3 Method", size=12.0), BODY) == 1
    assert heading_level(line("1. Introduction", size=14.0), BODY) == 1


def test_heading_level_detects_ieee_headings_without_a_font_signal():
    """IEEE sets sections at body size; only caps/italics distinguish them."""
    assert heading_level(line("I. INTRODUCTION"), BODY) == 1
    assert heading_level(line("III. DATA, NORMALISATION AND LABELLING"), BODY) == 1
    assert heading_level(line("A. Input Data", italic=True), BODY) == 2


def test_heading_level_rejects_body_text():
    prose = "The agent observes the limit order book and acts every 30 minutes."
    assert heading_level(line(prose), BODY) is None
    assert heading_level(line("we train the model on one year of quotes", bold=True), BODY) is None
    # An author initial is not a subsection heading.
    assert heading_level(line("A. Smith and B. Jones"), BODY) is None


def test_display_size_title_may_end_in_a_colon():
    title = "Universal features of price formation in financial markets:"
    assert heading_level(line(title, size=14.9), BODY) == 1
    assert heading_level(line("as shown in the table below:", size=BODY), BODY) is None


def test_drop_cap_is_distinguished_from_an_appendix_label_by_ratio():
    assert is_drop_cap(line("I", size=29.44), BODY) is True
    assert is_drop_cap(line("A", size=13.96), 10.5) is False
    assert is_drop_cap(line("IN", size=29.44), BODY) is False


def test_merge_split_headings_joins_a_wrapped_title():
    merged = merge_split_headings(
        [
            Element("DeepLOB: Deep Convolutional Neural Networks", 1, 1, 20.0, 100.0),
            Element("for Limit Order Books", 1, 1, 20.0, 124.0),
        ]
    )
    assert [e.text for e in merged] == [
        "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
    ]


def test_merge_split_headings_keeps_distinct_headings_apart():
    """Title / authors / "Abstract" are all level-1 on page 1; merging is wrong."""
    elements = [
        Element("Universal features of price formation", 1, 1, 14.9, 100.0),
        Element("Justin Sirignano and Rama Cont", 1, 1, 11.0, 160.0),  # other size
        Element("Abstract", 1, 1, 11.0, 400.0),  # far below
    ]
    assert [e.text for e in merge_split_headings(elements)] == [e.text for e in elements]


# --------------------------------------------------------------------------- #
# Layout / reading order
# --------------------------------------------------------------------------- #


def test_order_blocks_reads_left_column_before_right():
    left = [block(f"left{i}", x0=0.0, x1=45.0, y0=float(10 * i)) for i in range(3)]
    right = [block(f"right{i}", x0=55.0, x1=100.0, y0=float(10 * i)) for i in range(3)]
    ordered = order_blocks([b for pair in zip(left, right) for b in pair])
    assert [b.lines[0].text for b in ordered] == [
        "left0",
        "left1",
        "left2",
        "right0",
        "right1",
        "right2",
    ]


def test_order_blocks_treats_a_gutter_crossing_block_as_a_band_separator():
    """A centered title is only ~50% wide but belongs to neither column."""
    title = block("Centered Title", x0=25.0, x1=75.0, y0=0.0)
    left = [block(f"left{i}", x0=0.0, x1=45.0, y0=float(10 + 10 * i)) for i in range(2)]
    right = [block(f"right{i}", x0=55.0, x1=100.0, y0=float(10 + 10 * i)) for i in range(2)]
    ordered = order_blocks([*left, *right, title])
    assert [b.lines[0].text for b in ordered][0] == "Centered Title"
    assert [b.lines[0].text for b in ordered][1:3] == ["left0", "left1"]


def test_order_blocks_leaves_single_column_pages_in_y_order():
    blocks = [block(f"p{i}", x0=0.0, x1=100.0, y0=float(10 * i)) for i in range(4)]
    assert [b.lines[0].text for b in order_blocks(blocks[::-1])] == ["p0", "p1", "p2", "p3"]


def test_margin_stamp_is_detected():
    assert is_margin_stamp(block("arXiv:1706.10059v2", x0=0.0, x1=20.0, y0=0.0, y1=300.0))
    assert not is_margin_stamp(block("normal text", x0=0.0, x1=100.0, y0=0.0, y1=12.0))


# --------------------------------------------------------------------------- #
# Markdown assembly
# --------------------------------------------------------------------------- #


def test_render_sections_tile_the_document_exactly():
    text, sections = render(
        [
            Element("preamble text", 0),
            Element("1 Method", 1),
            Element("body of method", 0),
            Element("1.1 Reward", 2),
            Element("body of reward", 0),
        ]
    )
    assert [s.char_start for s in sections] == [0] + [
        s.char_start for s in sections if s.level
    ]
    # No gaps, no overlaps, full coverage.
    assert sections[0].char_start == 0
    assert sections[-1].char_end == len(text)
    for earlier, later in zip(sections, sections[1:]):
        assert earlier.char_end == later.char_start
    # Offsets address the real text.
    method = next(s for s in sections if s.title == "1 Method")
    assert text[method.char_start :].startswith("# 1 Method")


def test_render_builds_breadcrumbs_from_heading_levels():
    _, sections = render(
        [
            Element("3 Method", 1),
            Element("3.1 Reward", 2),
            Element("reward body", 0),
            Element("4 Results", 1),
            Element("results body", 0),
        ]
    )
    paths = {s.title: s.path for s in sections if s.level}
    assert paths["3.1 Reward"] == ("3 Method", "3.1 Reward")
    assert paths["4 Results"] == ("4 Results",)


# --------------------------------------------------------------------------- #
# Markdown notes
# --------------------------------------------------------------------------- #


def _note(doc_id="note_x", title="t"):
    return {"doc_id": doc_id, "title": title, "source_type": "note"}


def test_parse_markdown_drops_authoring_comments(tmp_path):
    path = tmp_path / "note_x.md"
    path.write_text(
        "# Title\n\n<!-- prompt: what market? -->\n\nReal prose here.\n", encoding="utf-8"
    )
    parsed = parse_markdown(path, _note())
    assert "prompt" not in parsed.text
    assert "Real prose here." in parsed.text
    assert parsed.parser == "markdown"
    assert [s.title for s in parsed.sections if s.level] == ["Title"]


def test_template_only_note_is_flagged(tmp_path):
    path = tmp_path / "note_x.md"
    path.write_text("# Scope\n\n<!-- what market? -->\n\n## State\n\n<!-- which features? -->\n")
    parsed = parse_document(_note(), tmp_path)
    assert parsed.stats["n_prose_chars"] == 0
    assert any(w.startswith("empty_or_template") for w in parsed.warnings)


def test_quality_warnings_catch_lost_word_spaces():
    stats = {
        "n_prose_chars": 5000,
        "chars_per_page": 2000.0,
        "mean_word_length": 12.07,
        "long_word_ratio": 0.187,
        "no_vowel_word_ratio": 0.01,
        "replacement_char_ratio": 0.0,
        "short_paragraph_ratio": 0.3,
        "n_sections": 10,
    }
    assert any(w.startswith("lost_word_spaces") for w in quality_warnings(stats, source_type="arxiv"))
    clean = {**stats, "mean_word_length": 5.1, "long_word_ratio": 0.0005}
    assert quality_warnings(clean, source_type="arxiv") == []


# --------------------------------------------------------------------------- #
# A real paper (skipped when the gitignored corpus is absent)
# --------------------------------------------------------------------------- #

REAL_DOC_ID = "arxiv_1706.10059"


@pytest.fixture
def real_paper():
    manifest = load_manifest()
    doc = next(
        (d for d in iter_documents(manifest) if d["doc_id"] == REAL_DOC_ID), None
    )
    if doc is None:
        pytest.skip(f"{REAL_DOC_ID} not in manifest")
    if not (DEFAULT_RAW_DIR / f"{REAL_DOC_ID}.pdf").exists():
        pytest.skip("corpus PDFs are gitignored; run `uv run quant-rag download` first")
    return parse_document(doc, DEFAULT_RAW_DIR)


def test_real_paper_parse_shape(real_paper):
    stats = real_paper.stats
    assert real_paper.n_pages > 10
    assert stats["n_chars"] > 50_000
    # Prose shape: spaces survived and the font-to-unicode mapping is sane.
    assert 4.0 < stats["mean_word_length"] < 7.0
    assert stats["long_word_ratio"] < 0.02
    assert stats["no_vowel_word_ratio"] < 0.12
    assert real_paper.warnings == []

    titles = [s.title for s in real_paper.sections if s.level]
    assert "Abstract" in titles
    assert any(t.startswith("1.") for t in titles)
    # Reading order: the title is whole and the abstract precedes the intro.
    assert titles[0] == (
        "A Deep Reinforcement Learning Framework for the "
        "Financial Portfolio Management Problem"
    )
    assert titles.index("Abstract") < titles.index("1. Introduction")

    # Sections tile the text, so every locator offset lands in exactly one.
    for earlier, later in zip(real_paper.sections, real_paper.sections[1:]):
        assert earlier.char_end == later.char_start
    assert real_paper.sections[-1].char_end == len(real_paper.text)

    # A golden-set locator phrase must survive parsing verbatim.
    assert "Ensemble of Identical Independent Evaluators" in real_paper.text


def test_parsed_metadata_round_trips(real_paper):
    meta = json.loads(json.dumps(real_paper.to_meta()))
    assert meta["doc_id"] == REAL_DOC_ID
    assert meta["parser"] == "pymupdf"
    assert len(meta["sections"]) == len(real_paper.sections)
