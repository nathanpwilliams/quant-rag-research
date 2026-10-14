"""Token counting for chunk sizing.

Chunking works on *character offsets*, not token strings: a chunk is a
``(char_start, char_end)`` slice of the parsed markdown. That keeps chunk text
byte-identical to the parsed document, which is what lets the eval's locator
resolver find a verbatim gold-evidence phrase inside a chunk. So a tokenizer
here only has to answer "where does each token start and end in this string".

``config.ingest.chunking.tokenizer`` names the embedder's tokenizer
(``BAAI/bge-base-en-v1.5``) because chunk_tokens should mean "tokens the
embedder will see". Loading it needs ``transformers``/``tokenizers``, which the
project has not taken on as a dependency yet, so:

* if the HF tokenizer is importable, it is used (exact);
* otherwise a stdlib regex tokenizer is used (approximate: it undercounts,
  since BERT WordPiece splits long/rare words that this treats as one token).

``Tokenizer.approximate`` records which happened, and the ingest report and the
chunk-set metadata both carry it, so no results CSV can claim exact BGE token
counts that were not actually measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Words (incl. digits/underscore) or single non-space symbols. Close enough to
# WordPiece for whole-corpus sizing; see module docstring for the caveat.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer(Protocol):
    name: str
    approximate: bool

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character ``(start, end)`` of every token, in order."""

    def count(self, text: str) -> int: ...


@dataclass
class RegexTokenizer:
    """Stdlib fallback: word/symbol tokens, no vocabulary."""

    name: str = "regex:word+symbol"
    approximate: bool = True

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN_RE.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in _TOKEN_RE.finditer(text))


@dataclass
class HFTokenizer:
    """Exact embedder tokenization via ``transformers`` offset mappings."""

    name: str
    approximate: bool = False
    _impl: object = None

    def spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._impl(  # type: ignore[operator]
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        # Special/empty tokens carry a (0, 0) mapping; drop them.
        return [(int(s), int(e)) for s, e in encoded["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        return len(self.spans(text))


def get_tokenizer(name: str, *, allow_download: bool = True) -> Tokenizer:
    """Return the HF tokenizer called ``name``, or the regex fallback.

    A missing dependency, no local cache, or no network all fall back rather
    than failing: corpus ingest must not be blocked on model downloads.
    """
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return RegexTokenizer()
    try:
        impl = AutoTokenizer.from_pretrained(name, local_files_only=not allow_download)
    except Exception:  # noqa: BLE001 - offline, missing cache, bad name: all fall back
        return RegexTokenizer()
    return HFTokenizer(name=name, _impl=impl)
