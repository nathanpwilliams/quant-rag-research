"""The embedding model, wrapped so the index and the retriever share one path.

Two rules this module exists to enforce:

* **Passages and queries are encoded asymmetrically.** BGE's model card
  prescribes an instruction prefix on queries and none on passages, so the two
  call sites are separate methods rather than one ``encode`` with a flag the
  caller can forget. ``encode_passages`` has no way to add an instruction.
* **The same model loads the same way at index time and at query time.** A
  query embedded by a differently-configured model than the corpus is a silent
  retrieval failure, not an error, so normalization and model name come from
  the one ``EmbeddingConfig`` the config file supplies.

Truncation is measured, not assumed: ``max_seq_length`` counts ``[CLS]`` and
``[SEP]``, so a chunk sized to exactly ``max_seq_length`` content tokens loses
its last two. :meth:`Embedder.truncation_report` quantifies that over a batch
so the index build can say so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quant_rag.config import EmbeddingConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

# [CLS] text [SEP]: the two special tokens BERT-family encoders add, which
# count against max_seq_length alongside the content tokens.
N_SPECIAL_TOKENS = 2


def resolve_device(requested: str) -> str:
    """Turn ``device: auto`` into a concrete device for this machine.

    Explicit values pass through untouched -- if the config says ``cuda`` on a
    machine with no CUDA, that should fail loudly rather than quietly run on
    the CPU at a tenth of the speed.
    """
    if requested != "auto":
        return requested
    import torch  # noqa: PLC0415 - heavyweight; only needed to probe devices

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TruncationReport:
    """How many texts the encoder had to cut, and by how much."""

    n_texts: int
    n_truncated: int
    max_tokens_dropped: int
    limit: int

    @property
    def ratio(self) -> float:
        return self.n_truncated / self.n_texts if self.n_texts else 0.0

    def __str__(self) -> str:
        if not self.n_truncated:
            return f"no truncation: every text fits in {self.limit} tokens"
        return (
            f"TRUNCATION: {self.n_truncated}/{self.n_texts} texts "
            f"({self.ratio:.0%}) exceed the model's {self.limit}-token window "
            f"and lose up to {self.max_tokens_dropped} token(s) off the end"
        )


class Embedder:
    """A loaded sentence-transformers model plus its config."""

    def __init__(self, config: EmbeddingConfig, *, model: SentenceTransformer | None = None):
        self.config = config
        self.device = resolve_device(config.device)
        if model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            model = SentenceTransformer(config.model, device=self.device)
        self.model = model

    @property
    def dim(self) -> int:
        # Renamed in sentence-transformers 6; support both so a version bump
        # does not silently change behaviour behind a deprecation warning.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    @property
    def metric(self) -> str:
        """What an inner product between two of these vectors means."""
        return "cosine" if self.config.normalize else "dot"

    def _encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            **kwargs,
        )
        # FAISS requires contiguous float32; float64 or a view raises at add().
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """Encode corpus text. No instruction prefix, by model-card rule."""
        return self._encode(texts, show_progress_bar=show_progress)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries, applying ``query_instruction`` if the config sets one."""
        prefix = self.config.query_instruction
        return self._encode([prefix + q for q in queries] if prefix else list(queries))

    def truncation_report(self, texts: list[str]) -> TruncationReport:
        """Count how many of ``texts`` the model's token window cuts off.

        Counted with the model's own tokenizer and ``truncation=False`` --
        asking the encoder would just return already-truncated lengths and
        report zero loss.
        """
        limit = self.max_seq_length
        budget = limit - N_SPECIAL_TOKENS
        tokenizer = self.model.tokenizer
        n_truncated = 0
        dropped = 0
        for text in texts:
            full = len(
                tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            )
            if full > budget:
                n_truncated += 1
                dropped = max(dropped, full - budget)
        return TruncationReport(len(texts), n_truncated, dropped, limit)
