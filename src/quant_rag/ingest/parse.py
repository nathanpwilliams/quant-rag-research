"""Parse corpus documents into markdown + per-document metadata.

Two source kinds, one output shape:

* **PDFs** (16 arXiv papers + the Bailey et al. preprint) go through PyMuPDF.
  Flat ``page.get_text()`` is not usable for two-column papers -- it reads
  across the gutter and interleaves the columns -- so this module works from
  the span dictionary instead: group spans into blocks, split each page into
  vertical bands at full-width elements, and read band-local columns
  left-then-right. Headings come from font size relative to the document's
  body size, which is also what gives the structure-aware chunker its sections.
* **Markdown notes** pass through with light normalization only (line endings,
  ligatures, trailing space, HTML comments dropped) -- no PyMuPDF.

Every ``ParsedDoc`` carries ``stats`` and ``warnings``. Parsing a 20-year-old
two-column PDF sometimes just fails, and a silently garbled document would
poison the index and quietly depress every ablation row, so suspicious parses
are flagged by measurement rather than discarded or hidden.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_rag.ingest.manifest import REPO_ROOT, Document

# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}
_LIGATURE_RE = re.compile("|".join(map(re.escape, LIGATURES)))
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
# IEEE two-column style: "III. DATA, NORMALISATION AND LABELLING" / "A. Input Data".
_ROMAN_HEADING_RE = re.compile(r"^[IVXLCDM]+\.\s+\S")
_LETTER_HEADING_RE = re.compile(r"^[A-Z]\.\s+\S")
# Page numbers: arabic, "3 / 12", or multi-letter roman front-matter numbering.
# Single roman letters are excluded on purpose -- a lone "I" is far more often
# a drop cap starting a section than a page label.
_PAGE_LABEL_RE = re.compile(
    r"^(?:page\s+)?(?:\d+(?:\s*/\s*\d+)?|[ivxlcdm]{2,})$", re.IGNORECASE
)
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")

# Section names that papers write without a number.
CANONICAL_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related work",
        "preliminaries",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "conclusions and future work",
        "future work",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "references",
        "bibliography",
        "appendix",
    }
)

BOLD_FLAG = 1 << 4
ITALIC_FLAG = 1 << 1
# Type1 fonts from TeX (CMBX10 = Computer Modern Bold Extended) often ship
# without the descriptor bits PyMuPDF reads, so the name is the only signal.
BOLD_FONT_MARKERS = ("bold", "black", "heavy", "semib", "-bd", "bx")
ITALIC_FONT_MARKERS = ("ital", "oblique")


def font_is_bold(font: str, flags: int) -> bool:
    name = font.lower()
    return bool(flags & BOLD_FLAG) or any(m in name for m in BOLD_FONT_MARKERS)


def font_is_italic(font: str, flags: int) -> bool:
    name = font.lower()
    return bool(flags & ITALIC_FLAG) or any(m in name for m in ITALIC_FONT_MARKERS)


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0


def normalize_text(text: str) -> str:
    """Ligatures -> ASCII, drop control chars, normalize line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LIGATURE_RE.sub(lambda m: LIGATURES[m.group(0)], text)
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("‐", "-").replace("‑", "-")
    return _CONTROL_RE.sub("", text)


def join_wrapped_lines(lines: list[str]) -> str:
    """Join visual line wraps into one paragraph, undoing hyphenation.

    ``"regu-"`` + ``"larization"`` -> ``"regularization"``, but ``"risk-"`` +
    ``"Adjusted"`` keeps its hyphen: a capital after the break usually means a
    real compound rather than a wrap.
    """
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


# --------------------------------------------------------------------------- #
# Parsed representation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Section:
    """A heading and the text under it, as offsets into ``ParsedDoc.text``.

    ``path`` is the heading breadcrumb (``("3 Method", "3.2 Reward")``); the
    structure-aware chunker prepends it so a chunk carries its own context.
    Offsets, not substrings, so chunk text stays identical to the parsed file.
    """

    title: str
    level: int
    path: tuple[str, ...]
    char_start: int
    char_end: int
    page_start: int | None = None

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    source_type: str
    source_file: str
    parser: str
    text: str
    sections: list[Section] = field(default_factory=list)
    n_pages: int | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def section_at(self, offset: int) -> Section | None:
        for section in self.sections:
            if section.char_start <= offset < section.char_end:
                return section
        return None

    def to_meta(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "parser": self.parser,
            "n_pages": self.n_pages,
            "n_chars": len(self.text),
            "sections": [
                {
                    "title": s.title,
                    "level": s.level,
                    "path": list(s.path),
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "page_start": s.page_start,
                }
                for s in self.sections
            ],
            "stats": self.stats,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Markdown assembly (shared by both source kinds)
# --------------------------------------------------------------------------- #


@dataclass
class Element:
    """One markdown block: a heading (``level >= 1``) or a paragraph.

    ``size``/``y0`` are the source line's font size and vertical position, kept
    only for PDF headings so :func:`merge_split_headings` can tell a wrapped
    heading from two different headings that happen to be adjacent.
    """

    text: str
    level: int = 0
    page: int | None = None
    size: float | None = None
    y0: float | None = None


def render(elements: list[Element]) -> tuple[str, list[Section]]:
    """Render elements to markdown and derive sections from heading offsets.

    A section runs from its heading to the next heading of *any* level, so
    sections tile the document exactly and every character belongs to one.
    """
    parts: list[str] = []
    offset = 0
    heads: list[tuple[int, Element, int]] = []  # (char_start, element, body_start)

    for element in elements:
        if not element.text:
            continue
        rendered = (
            f"{'#' * min(element.level, 6)} {element.text}" if element.level else element.text
        )
        if parts:
            parts.append("\n\n")
            offset += 2
        start = offset
        parts.append(rendered)
        offset += len(rendered)
        if element.level:
            heads.append((start, element, offset))

    text = "".join(parts)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    if heads and heads[0][0] > 0:
        sections.append(
            Section(title="", level=0, path=(), char_start=0, char_end=heads[0][0])
        )
    elif not heads and text:
        sections.append(Section(title="", level=0, path=(), char_start=0, char_end=len(text)))

    for i, (start, element, _) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        while stack and stack[-1][0] >= element.level:
            stack.pop()
        stack.append((element.level, element.text))
        sections.append(
            Section(
                title=element.text,
                level=element.level,
                path=tuple(title for _, title in stack),
                char_start=start,
                char_end=end,
                page_start=element.page,
            )
        )
    return text, sections


# --------------------------------------------------------------------------- #
# Markdown notes
# --------------------------------------------------------------------------- #


def parse_markdown(path: Path, doc: Document) -> ParsedDoc:
    """Light normalization only: the notes are already markdown."""
    raw = normalize_text(path.read_text(encoding="utf-8"))
    # The note templates carry their prompts in HTML comments. They are
    # authoring scaffolding, not corpus content, and indexing them would put
    # question-shaped text in the retrieval pool.
    body = _HTML_COMMENT_RE.sub("", raw)
    elements: list[Element] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            elements.append(Element(text="\n".join(buffer).strip()))
            buffer.clear()

    for line in body.split("\n"):
        heading = _ATX_HEADING_RE.match(line)
        if heading:
            flush()
            elements.append(Element(text=heading.group(2), level=len(heading.group(1))))
        elif line.strip():
            buffer.append(line.rstrip())
        else:
            flush()
    flush()

    text, sections = render(elements)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return ParsedDoc(
        doc_id=doc["doc_id"],
        title=doc["title"],
        source_type=doc["source_type"],
        source_file=_relative(path),
        parser="markdown",
        text=text,
        sections=sections,
        n_pages=None,
    )


# --------------------------------------------------------------------------- #
# PDF layout model
# --------------------------------------------------------------------------- #


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    italic: bool
    x0: float
    x1: float
    y0: float


@dataclass
class Block:
    lines: list[Line]
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2


def page_blocks(page: Any) -> list[Block]:
    blocks: list[Block] = []
    for raw in page.get_text("dict").get("blocks", []):
        if raw.get("type") != 0:  # 0 = text; images/drawings carry no text
            continue
        lines: list[Line] = []
        for raw_line in raw.get("lines", []):
            spans = raw_line.get("spans", [])
            # Join *all* spans: in LaTeX-produced PDFs an inter-word space is
            # often its own span (the space glyph comes from another font), so
            # dropping whitespace-only spans glues words together.
            text = normalize_text("".join(s.get("text", "") for s in spans)).strip()
            if not text:
                continue
            # ...but font size and weight must come from the spans that have ink.
            inked = [s for s in spans if s.get("text", "").strip()]
            if not inked:
                continue
            lines.append(
                Line(
                    text=text,
                    size=max(float(s.get("size", 0.0)) for s in inked),
                    bold=all(
                        font_is_bold(s.get("font", ""), int(s.get("flags", 0))) for s in inked
                    ),
                    italic=all(
                        font_is_italic(s.get("font", ""), int(s.get("flags", 0)))
                        for s in inked
                    ),
                    x0=raw_line["bbox"][0],
                    x1=raw_line["bbox"][2],
                    y0=raw_line["bbox"][1],
                )
            )
        if lines:
            x0, y0, x1, y1 = raw["bbox"]
            blocks.append(Block(lines=lines, x0=x0, y0=y0, x1=x1, y1=y1))
    return blocks


def is_margin_stamp(block: Block) -> bool:
    """arXiv prints a rotated ``arXiv:1706.10059v2 [q-fin.CP]`` stamp down the
    left margin; it lands as a tall, very narrow block of unrelated text."""
    return block.width < 30 and block.height > 100


def order_blocks(blocks: list[Block]) -> list[Block]:
    """Put a page's blocks into reading order, column-aware.

    Full-width blocks (title, abstract, a table spanning the gutter) split the
    page into vertical bands. Inside a band, if blocks clearly separate into a
    left and a right group, read the whole left column before the right one --
    sorting by ``y`` alone is what produces interleaved-column garbage.
    """
    blocks = [b for b in blocks if not is_margin_stamp(b)]
    if len(blocks) < 2:
        return blocks

    left_edge = min(b.x0 for b in blocks)
    right_edge = max(b.x1 for b in blocks)
    content_width = right_edge - left_edge
    mid = (left_edge + right_edge) / 2

    # In a real two-column layout nothing in the body crosses the gutter, so
    # *crossing* is the test -- not width. A centered title line is only ~50%
    # of the content width but still belongs to neither column, and treating
    # it as a right-column block pushes it after the whole left column.
    tolerance = 0.02 * content_width

    def spans_gutter(block: Block) -> bool:
        return block.x0 < mid - tolerance and block.x1 > mid + tolerance

    bands: list[list[Block]] = []
    current: list[Block] = []
    for block in sorted(blocks, key=lambda b: b.y0):
        if spans_gutter(block):
            if current:
                bands.append(current)
                current = []
            bands.append([block])
        else:
            current.append(block)
    if current:
        bands.append(current)

    ordered: list[Block] = []
    for band in bands:
        left = [b for b in band if b.center_x < mid]
        right = [b for b in band if b.center_x >= mid]
        # Require real evidence of two columns; otherwise a couple of short
        # centered lines would get reordered for no reason.
        if len(left) >= 2 and len(right) >= 2:
            ordered.extend(sorted(left, key=lambda b: b.y0))
            ordered.extend(sorted(right, key=lambda b: b.y0))
        else:
            ordered.extend(sorted(band, key=lambda b: (b.y0, b.x0)))
    return ordered


def body_font_size(pages: list[list[Block]]) -> float:
    """The most common font size by character count: the body text size."""
    weights: Counter[float] = Counter()
    for blocks in pages:
        for block in blocks:
            for line in block.lines:
                weights[round(line.size * 2) / 2] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def heading_level(line: Line, body_size: float) -> int | None:
    """Return a heading level for ``line``, or ``None`` if it is body text.

    Font size alone is not enough. IEEE two-column papers set section headings
    at the *body* size, distinguished only by small caps (``I. INTRODUCTION``)
    and their subsections only by italics (``A. Input Data``), so both a font
    signal and a label-shape signal are needed.
    """
    text = line.text.strip()
    if not text or len(text) > 120 or len(text.split()) > 14:
        return None

    numbered = _NUMBERED_HEADING_RE.match(text)
    # Headings do not read as sentences. Exempt numbered headings ("3.
    # Method.") and display-size lines: a title set at 1.5x body size
    # legitimately ends in a colon ("Universal features of price formation in
    # financial markets:").
    if (
        text.endswith((".", ",", ";", ":"))
        and not numbered
        and line.size < body_size + 2
    ):
        return None

    # Label-shaped headings, independent of font metrics.
    if _ROMAN_HEADING_RE.match(text) and uppercase_ratio(text) >= 0.6:
        return 1
    if _LETTER_HEADING_RE.match(text) and (line.italic or uppercase_ratio(text) >= 0.6):
        return 2

    bigger = line.size >= body_size + 0.6
    if not (bigger or line.bold):
        return None
    if numbered:
        return min(numbered.group(1).count(".") + 1, 6)
    if text.lower().rstrip(".").strip() in CANONICAL_HEADINGS:
        return 1
    if bigger and line.size >= body_size + 2:
        return 1
    if bigger:
        return 2
    return None


DROP_CAP_SIZE_RATIO = 1.8


def is_drop_cap(line: Line, body_size: float) -> bool:
    """An oversized single capital opening a section ("**I**N today's ...").

    The size *ratio* matters, not the difference: a drop cap runs 2-3x body
    size (DeepLOB's is 2.94x), whereas a lone appendix label set as a heading
    ("A", then "Hyperparameters") is only ~1.3x and must stay a heading --
    treating it as a drop cap glues it onto the next paragraph.
    """
    text = line.text.strip()
    return (
        len(text) == 1 and text.isupper() and line.size >= DROP_CAP_SIZE_RATIO * body_size
    )


def merge_split_headings(elements: list[Element]) -> list[Element]:
    """Join adjacent heading lines that are one heading broken across lines.

    A title wrapped over two lines, or an appendix label and its name set on
    separate lines ("A" / "Hyperparameters"), arrive as two heading elements of
    the same level on the same page with no body text between them.

    Same level is not sufficient: on a title page the title, the author line
    and "Abstract" can all qualify, and merging them produces one nonsense
    heading. A wrapped heading is also set in the same size and sits on the
    next line down, so both are required.
    """
    merged: list[Element] = []
    for element in elements:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and element.level
            and previous.level == element.level
            and previous.page == element.page
            and len(previous.text) + len(element.text) <= 200
            and previous.size is not None
            and element.size is not None
            and abs(previous.size - element.size) < 0.5
            and previous.y0 is not None
            and element.y0 is not None
            and 0 <= element.y0 - previous.y0 <= 2.2 * previous.size
        ):
            merged[-1] = Element(
                text=f"{previous.text} {element.text}",
                level=element.level,
                page=element.page,
                size=element.size,
                y0=element.y0,
            )
        else:
            merged.append(element)
    return merged


def _running_head_texts(pages: list[list[Block]]) -> set[str]:
    """Normalized first/last-line texts that repeat across most pages."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for blocks in pages:
        candidates = []
        if blocks:
            candidates.append(blocks[0].lines[0].text)
            candidates.append(blocks[-1].lines[-1].text)
        for text in candidates:
            counts[re.sub(r"\d+", "#", text.strip().lower())] += 1
    threshold = max(3, int(0.5 * len(pages)))
    return {text for text, n in counts.items() if n >= threshold and text}


def parse_pdf(path: Path, doc: Document, *, parser: str = "pymupdf") -> ParsedDoc:
    import pymupdf  # noqa: PLC0415 - heavyweight; keep import local to the call

    elements: list[Element] = []
    with pymupdf.open(path) as pdf:
        n_pages = pdf.page_count
        pages = [page_blocks(page) for page in pdf]

    body_size = body_font_size(pages)
    running = _running_head_texts(pages)

    for page_no, blocks in enumerate(pages, start=1):
        # A drop cap is its own line, sometimes its own block, so the carry
        # has to outlive the block loop.
        drop_cap = ""

        for block in order_blocks(blocks):
            pending: list[str] = []

            def flush() -> None:
                paragraph = join_wrapped_lines(pending)
                pending.clear()
                if paragraph:
                    elements.append(Element(text=paragraph, page=page_no))

            for line in block.lines:
                normalized = re.sub(r"\d+", "#", line.text.strip().lower())
                if normalized in running or _PAGE_LABEL_RE.match(line.text.strip()):
                    continue
                if is_drop_cap(line, body_size):
                    drop_cap = line.text.strip()
                    continue
                level = heading_level(line, body_size)
                if level is not None:
                    flush()
                    # A carry that runs into a heading belongs to the heading
                    # (an appendix label on its own line); never silently drop it.
                    title = line.text.strip()
                    if drop_cap:
                        title = f"{drop_cap} {title}"
                        drop_cap = ""
                    elements.append(
                        Element(
                            text=title,
                            level=level,
                            page=page_no,
                            size=line.size,
                            y0=line.y0,
                        )
                    )
                else:
                    # Re-attach the drop cap to the word it belongs to.
                    pending.append(drop_cap + line.text if drop_cap else line.text)
                    drop_cap = ""
            flush()

    text, sections = render(merge_split_headings(elements))
    return ParsedDoc(
        doc_id=doc["doc_id"],
        title=doc["title"],
        source_type=doc["source_type"],
        source_file=_relative(path),
        parser=parser,
        text=text,
        sections=sections,
        n_pages=n_pages,
    )


# --------------------------------------------------------------------------- #
# Parse-quality measurement
# --------------------------------------------------------------------------- #

MIN_CHARS_PER_PAGE = 400
MAX_NO_VOWEL_RATIO = 0.12
MAX_REPLACEMENT_RATIO = 0.005
MIN_PROSE_CHARS = 400
# English prose runs ~4.5-5.5 chars/word; a clean parse of this corpus measures
# 4.9-5.8. A PDF whose inter-word spaces were lost measures 10-15 with ~20% of
# "words" over 20 chars, which is how the span-join bug was caught -- hence
# these two, rather than a general-purpose "looks wrong" score.
MAX_MEAN_WORD_LENGTH = 8.0
MAX_LONG_WORD_RATIO = 0.02
# Formula-heavy papers legitimately parse into many short blocks (one per
# display equation), so this only fires when almost nothing is prose-shaped.
MAX_SHORT_PARAGRAPH_RATIO = 0.80


def quality_stats(doc: ParsedDoc) -> dict[str, Any]:
    text = doc.text
    paragraphs = [p for p in text.split("\n\n") if p and not p.startswith("#")]
    words = _WORD_RE.findall(text)
    long_words = [w for w in words if len(w) >= 3]
    no_vowel = sum(1 for w in long_words if not _VOWEL_RE.search(w))
    glued = sum(1 for w in words if len(w) >= 20)
    short_paragraphs = sum(1 for p in paragraphs if len(p.split()) < 5)
    prose_chars = sum(len(p) for p in paragraphs)

    return {
        "n_chars": len(text),
        "n_prose_chars": prose_chars,
        "n_words": len(words),
        "n_sections": sum(1 for s in doc.sections if s.level),
        "n_paragraphs": len(paragraphs),
        "chars_per_page": round(len(text) / doc.n_pages, 1) if doc.n_pages else None,
        "mean_word_length": round(sum(map(len, words)) / len(words), 2) if words else 0.0,
        "long_word_ratio": round(glued / len(words), 4) if words else 0.0,
        "no_vowel_word_ratio": round(no_vowel / len(long_words), 4) if long_words else 0.0,
        "replacement_char_ratio": round(text.count("�") / len(text), 5) if text else 0.0,
        "short_paragraph_ratio": (
            round(short_paragraphs / len(paragraphs), 3) if paragraphs else 1.0
        ),
        "mean_words_per_paragraph": (
            round(sum(len(p.split()) for p in paragraphs) / len(paragraphs), 1)
            if paragraphs
            else 0.0
        ),
    }


def quality_warnings(stats: dict[str, Any], *, source_type: str) -> list[str]:
    """Flag parses that should not be trusted in the index without a look."""
    warnings: list[str] = []
    if stats["n_prose_chars"] < MIN_PROSE_CHARS:
        warnings.append(
            f"empty_or_template: only {stats['n_prose_chars']} chars of prose outside headings"
        )
    cpp = stats["chars_per_page"]
    if cpp is not None and cpp < MIN_CHARS_PER_PAGE:
        warnings.append(
            f"low_text_yield: {cpp} chars/page (scanned or image-only PDF?)"
        )
    if (
        stats["mean_word_length"] > MAX_MEAN_WORD_LENGTH
        or stats["long_word_ratio"] > MAX_LONG_WORD_RATIO
    ):
        warnings.append(
            f"lost_word_spaces: mean word length {stats['mean_word_length']}, "
            f"{stats['long_word_ratio']:.1%} of words over 20 chars "
            "(PDF encodes inter-word gaps positionally?)"
        )
    if stats["no_vowel_word_ratio"] > MAX_NO_VOWEL_RATIO:
        warnings.append(
            f"garbled_encoding: {stats['no_vowel_word_ratio']:.1%} of words have no vowel "
            "(broken font-to-unicode mapping?)"
        )
    if stats["replacement_char_ratio"] > MAX_REPLACEMENT_RATIO:
        warnings.append(
            f"undecodable_chars: {stats['replacement_char_ratio']:.2%} U+FFFD"
        )
    if stats["short_paragraph_ratio"] > MAX_SHORT_PARAGRAPH_RATIO:
        warnings.append(
            f"fragmented_layout: {stats['short_paragraph_ratio']:.0%} of paragraphs are "
            "under 5 words (figure/table soup or broken reading order?)"
        )
    if source_type != "note" and stats["n_sections"] == 0:
        warnings.append("no_headings_detected: structure-aware chunking will see one section")
    return warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_path(doc: Document, raw_dir: Path) -> Path:
    """Where the document's bytes live, from the manifest's ``file`` if set."""
    file = doc.get("file")
    if file:
        candidate = Path(file)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate
    suffix = ".md" if doc.get("source_type") == "note" else ".pdf"
    return raw_dir / f"{doc['doc_id']}{suffix}"


def parse_document(doc: Document, raw_dir: Path, *, parser: str = "pymupdf") -> ParsedDoc:
    """Parse one manifest document, measuring and flagging parse quality."""
    path = source_path(doc, raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{doc['doc_id']}: source file not found: {path}")

    if path.suffix.lower() == ".md":
        parsed = parse_markdown(path, doc)
    elif parser == "pymupdf":
        parsed = parse_pdf(path, doc, parser=parser)
    else:
        raise NotImplementedError(
            f"ingest.parser={parser!r} is not implemented; "
            "'pymupdf4llm' needs the pymupdf4llm dependency (see CLAUDE.md rule 5)"
        )

    parsed.stats = quality_stats(parsed)
    parsed.warnings = quality_warnings(parsed.stats, source_type=parsed.source_type)
    return parsed
