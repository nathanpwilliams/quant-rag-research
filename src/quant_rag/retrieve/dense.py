"""Dense retrieval: embed the query, exact inner-product search over the index.

This is the baseline ablation row's whole retrieval path. It is deliberately
thin — there is no fusion, no rerank, no score transformation — because every
later row is measured against it, and a baseline that quietly massaged scores
would make those deltas unreadable.

Scoring, stated exactly:

* The score is the raw inner product ``q · d`` between the query vector and
  each chunk vector, sorted descending. Both backends
  (:mod:`quant_rag.index.backends`) are exhaustive: every chunk is scored, so
  the top-k is the true top-k, never an approximation.
* With ``index.embedding.normalize: true`` both sides are L2-normalized, so
  ``q · d`` *is* cosine similarity, in ``[-1, 1]`` (in practice ~0.5-0.9 for
  BGE on topical matches — BGE's absolute similarities run high, so only the
  ordering and the gaps are meaningful, not the level).
* The score is reported exactly as the backend computes it. No min-max
  scaling, no softmax, no length normalization. ``metric`` in ``index.json``
  records which of cosine/dot the numbers are, so a results CSV can never be
  ambiguous about it.
* Ties go to the lower chunk ordinal, deterministically, so a rerun cannot
  reshuffle equal-scoring chunks and move an eval number.

``k`` is a parameter rather than being read from the config inside this class:
the eval sweeps ``k`` over ``eval.k_values`` against one retriever, so the
config's ``retrieve.top_k`` is applied by the caller (see ``ask``), not baked
in here.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_rag.config import PipelineConfig
from quant_rag.index.embed import Embedder
from quant_rag.index.store import LoadedIndex, load_index
from quant_rag.ingest.chunk import Chunk


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and its score, at a rank."""

    rank: int  # 1-based, as reported
    score: float
    chunk: Chunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class DenseRetriever:
    """Query-embedding + exact top-k over one verified index/chunkset pair."""

    mode = "dense"

    def __init__(self, index: LoadedIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder
        # The query vector must come from the same model that built the index,
        # or the scores are meaningless in a way nothing downstream would
        # notice. The index metadata is the record of what actually built it.
        if embedder.config.model != index.meta.model:
            raise ValueError(
                f"embedder model {embedder.config.model!r} does not match the "
                f"index's {index.meta.model!r}"
            )
        if embedder.config.normalize != index.meta.normalize:
            raise ValueError(
                f"embedder normalize={embedder.config.normalize} does not match "
                f"the index's normalize={index.meta.normalize}"
            )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> DenseRetriever:
        index = load_index(config)
        return cls(index, Embedder(config.index.embedding))

    @property
    def metric(self) -> str:
        return self.index.meta.metric

    @property
    def backend(self) -> str:
        return self.index.backend.name

    def search(self, query: str, k: int) -> list[Hit]:
        """Top-``k`` chunks for ``query``, highest score first.

        Fewer than ``k`` hits come back when the corpus is smaller than ``k``:
        backends clamp the width to the corpus size rather than padding, so a
        sentinel row never becomes a bogus chunk.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query_vector = self.embedder.encode_queries([query])
        scores, rows = self.index.search(query_vector, k)

        hits: list[Hit] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:  # defensive: a padded row should never reach here
                continue
            hits.append(
                Hit(rank=len(hits) + 1, score=float(score), chunk=self.index.chunk_at(int(row)))
            )
        return hits


def build_retriever(config: PipelineConfig) -> DenseRetriever:
    """Build the retriever named by ``retrieve.mode``.

    Only ``dense`` exists so far. ``sparse`` and ``hybrid`` are config-valid
    but unimplemented, and must fail loudly: silently falling back to dense
    would publish an ablation row that did not run the retrieval it names.
    """
    if config.retrieve.mode != "dense":
        raise NotImplementedError(
            f"retrieve.mode={config.retrieve.mode!r} is not implemented yet; "
            "only 'dense' is available"
        )
    return DenseRetriever.from_config(config)
