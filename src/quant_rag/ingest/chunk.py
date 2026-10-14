"""Chunkers: ``fixed`` and ``structure``, behind one interface.

Both produce ``Chunk`` records addressed by character offsets into
``ParsedDoc.text``. That is deliberate: the eval resolves a golden-set locator
(a verbatim phrase) to chunk IDs under whatever chunking config is active, so a
chunk has to be an exact slice of the parsed document, and its ID has to be
reproducible from the document plus the config -- never from iteration order of
a dict or a content hash that moves when a neighbouring chunk changes.

``chunk_id`` is ``<doc_id>#<ordinal>``, ordinal being the chunk's position in
the document. Two chunking configs give the same IDs to different spans, so
chunk sets are stored per config fingerprint (see ``config.ingest_fingerprint``)
and every chunk record carries the knobs it was built with.

``fixed``
    Sliding window of ``chunk_tokens`` with ``overlap_tokens`` of carry-over,
    straight over the document text. Structure-blind; the baseline row.
``structure``
    Splits on detected section headings, packs consecutive sections up to
    ``chunk_tokens`` without crossing a top-level section boundary, windows any
    single section that is too big, and prepends the heading breadcrumb to each
    chunk so an isolated chunk still says what part of the paper it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from quant_rag.config import ChunkingConfig
from quant_rag.ingest.parse import ParsedDoc, Section
from quant_rag.ingest.tokenizer import Tokenizer, get_tokenizer


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    ``text`` is what gets embedded, which for the structure chunker includes
    the ``prefix`` breadcrumb. ``body`` is the verbatim document slice
    ``doc.text[char_start:char_end]`` -- that is the half a locator must match.
    """

    chunk_id: str
    doc_id: str
    index: int
    text: str
    char_start: int
    char_end: int
    n_tokens: int
    strategy: str
    prefix: str = ""
    section_title: str = ""
    section_path: tuple[str, ...] = ()
    page_start: int | None = None

    @property
    def body(self) -> str:
        return self.text[len(self.prefix) :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "index": self.index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "n_tokens": self.n_tokens,
            "strategy": self.strategy,
            "prefix": self.prefix,
            "section_title": self.section_title,
            "section_path": list(self.section_path),
            "page_start": self.page_start,
        }


def chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#{index:04d}"


class Chunker(Protocol):
    strategy: str

    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...


# --------------------------------------------------------------------------- #
# Shared windowing
# --------------------------------------------------------------------------- #


def window_indices(
    n_tokens: int, chunk_tokens: int, overlap_tokens: int
) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` token index windows covering ``n_tokens``.

    The last window is short rather than padded, and is never emitted twice:
    once a window reaches the end, iteration stops. ``ChunkingConfig`` already
    guarantees ``overlap_tokens < chunk_tokens``, so the stride is positive.
    """
    if n_tokens <= 0:
        return []
    stride = chunk_tokens - overlap_tokens
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_tokens, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += stride
    return windows


def _trim(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Shrink ``[lo, hi)`` off surrounding whitespace, keeping offsets exact."""
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def _body_span(doc: ParsedDoc, section: Section) -> tuple[int, int]:
    """A section's text without its own rendered heading line."""
    start = section.char_start
    if section.level:
        start += min(section.level, 6) + 1 + len(section.title)
    return _trim(doc.text, start, section.char_end)


# --------------------------------------------------------------------------- #
# fixed
# --------------------------------------------------------------------------- #


@dataclass
class FixedChunker:
    """Structure-blind sliding window over the whole document."""

    chunk_tokens: int
    overlap_tokens: int
    tokenizer: Tokenizer
    strategy: str = "fixed"

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        spans = self.tokenizer.spans(doc.text)
        chunks: list[Chunk] = []
        for index, (first, last) in enumerate(
            window_indices(len(spans), self.chunk_tokens, self.overlap_tokens)
        ):
            lo, hi = _trim(doc.text, spans[first][0], spans[last - 1][1])
            if lo >= hi:
                continue
            section = doc.section_at(lo)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(doc.doc_id, index),
                    doc_id=doc.doc_id,
                    index=index,
                    text=doc.text[lo:hi],
                    char_start=lo,
                    char_end=hi,
                    n_tokens=last - first,
                    strategy=self.strategy,
                    section_title=section.title if section else "",
                    section_path=section.path if section else (),
                    page_start=section.page_start if section else None,
                )
            )
        return chunks


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


@dataclass
class _Unit:
    """One section's body, with the tokens it costs."""

    section: Section
    lo: int
    hi: int
    n_tokens: int

    @property
    def root(self) -> str:
        """Top-level section this belongs to; packing never crosses roots."""
        return self.section.path[0] if self.section.path else ""


@dataclass
class StructureChunker:
    """Section-aware: pack small sections, window big ones, prepend headings."""

    chunk_tokens: int
    overlap_tokens: int
    tokenizer: Tokenizer
    strategy: str = "structure"

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        units: list[_Unit] = []
        for section in doc.sections:
            lo, hi = _body_span(doc, section)
            if lo >= hi:
                continue  # heading with no body of its own; it lives on in `path`
            units.append(
                _Unit(
                    section=section,
                    lo=lo,
                    hi=hi,
                    n_tokens=self.tokenizer.count(doc.text[lo:hi]),
                )
            )

        chunks: list[Chunk] = []
        for group in self._pack(units):
            if len(group) == 1 and group[0].n_tokens > self.chunk_tokens:
                chunks.extend(self._split_unit(doc, group[0], len(chunks)))
            else:
                chunks.append(self._emit(doc, group, len(chunks)))
        return chunks

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Greedily group consecutive sections, respecting roots and budget."""
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        budget = 0
        for unit in units:
            oversized = unit.n_tokens > self.chunk_tokens
            new_root = bool(current) and unit.root != current[-1].root
            if current and (oversized or new_root or budget + unit.n_tokens > self.chunk_tokens):
                groups.append(current)
                current, budget = [], 0
            if oversized:
                groups.append([unit])
                continue
            current.append(unit)
            budget += unit.n_tokens
        if current:
            groups.append(current)
        return groups

    def _prefix(self, group: list[_Unit]) -> tuple[str, Section]:
        """Breadcrumb shared by every section in the group, plus its anchor."""
        anchor = group[0].section
        path = anchor.path
        for unit in group[1:]:
            other = unit.section.path
            common = 0
            while common < min(len(path), len(other)) and path[common] == other[common]:
                common += 1
            path = path[:common]
        label = " > ".join(path) if path else anchor.breadcrumb
        return (f"{label}\n\n" if label else ""), anchor

    def _emit(self, doc: ParsedDoc, group: list[_Unit], index: int) -> Chunk:
        prefix, anchor = self._prefix(group)
        lo, hi = _trim(doc.text, group[0].lo, group[-1].hi)
        body = doc.text[lo:hi]
        return Chunk(
            chunk_id=chunk_id(doc.doc_id, index),
            doc_id=doc.doc_id,
            index=index,
            text=prefix + body,
            char_start=lo,
            char_end=hi,
            n_tokens=self.tokenizer.count(prefix + body),
            strategy=self.strategy,
            prefix=prefix,
            section_title=anchor.title,
            section_path=anchor.path,
            page_start=anchor.page_start,
        )

    def _split_unit(self, doc: ParsedDoc, unit: _Unit, start_index: int) -> list[Chunk]:
        """Window one over-long section, repeating the breadcrumb on each part."""
        prefix, anchor = self._prefix([unit])
        spans = self.tokenizer.spans(doc.text[unit.lo : unit.hi])
        # Leave room for the prefix so the embedded chunk still fits the budget.
        budget = max(self.chunk_tokens - self.tokenizer.count(prefix), 1)
        overlap = min(self.overlap_tokens, budget - 1) if budget > 1 else 0
        chunks: list[Chunk] = []
        for offset, (first, last) in enumerate(window_indices(len(spans), budget, overlap)):
            lo, hi = _trim(
                doc.text, unit.lo + spans[first][0], unit.lo + spans[last - 1][1]
            )
            if lo >= hi:
                continue
            index = start_index + offset
            body = doc.text[lo:hi]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(doc.doc_id, index),
                    doc_id=doc.doc_id,
                    index=index,
                    text=prefix + body,
                    char_start=lo,
                    char_end=hi,
                    n_tokens=self.tokenizer.count(prefix + body),
                    strategy=self.strategy,
                    prefix=prefix,
                    section_title=anchor.title,
                    section_path=anchor.path,
                    page_start=anchor.page_start,
                )
            )
        return chunks


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

CHUNKERS = {"fixed": FixedChunker, "structure": StructureChunker}


def build_chunker(cfg: ChunkingConfig, tokenizer: Tokenizer | None = None) -> Chunker:
    """Build the chunker named by the config. No default strategy, by design:
    the active strategy always comes from the config file."""
    try:
        factory = CHUNKERS[cfg.strategy]
    except KeyError:  # pragma: no cover - ChunkingConfig validates first
        raise ValueError(f"unknown chunking strategy: {cfg.strategy!r}") from None
    return factory(
        chunk_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.overlap_tokens,
        tokenizer=tokenizer if tokenizer is not None else get_tokenizer(cfg.tokenizer),
    )
