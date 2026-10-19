"""Index tests.

The expensive part of this stage is the embedding model, and what needs testing
is not BGE but the bookkeeping around it: that an index is bound to the chunk
set it was built from, that it survives a round trip to disk unchanged, and
that a query lands on the chunk a human would pick. So most tests here drive
``build_index``/``load_index`` with a tiny deterministic stand-in embedder, and
only the retrieval-ranking test loads the real model (skipped if unavailable).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from quant_rag.config import (
    ChunkingConfig,
    ConfigError,
    CorpusConfig,
    EmbeddingConfig,
    EvalConfig,
    ExperimentConfig,
    FaissConfig,
    FusionConfig,
    GenerateConfig,
    IndexConfig,
    IngestConfig,
    PipelineConfig,
    RagasConfig,
    RerankConfig,
    RetrieveConfig,
    BM25Config,
)
from quant_rag.index import store
from quant_rag.index.embed import N_SPECIAL_TOKENS, resolve_device
from quant_rag.index.store import (
    IndexMissingError,
    IndexPairingError,
    build_index,
    index_dir,
    load_index,
)
from quant_rag.ingest.chunk import Chunk

DIM = 16


# --------------------------------------------------------------------------- #
# Fixtures: a config pointing at a temp dir, and a deterministic embedder
# --------------------------------------------------------------------------- #


def make_config(
    tmp_path,
    *,
    chunk_tokens: int = 64,
    model: str = "fake-model",
    normalize: bool = True,
    backend: str = "exact_numpy",
    query_instruction: str = "Q: ",
    mode: str = "dense",
    top_k: int = 10,
) -> PipelineConfig:
    return PipelineConfig(
        experiment=ExperimentConfig(name="test", description="d"),
        corpus=CorpusConfig(
            manifest=tmp_path / "manifest.json",
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        ),
        ingest=IngestConfig(
            parser="pymupdf",
            chunking=ChunkingConfig(
                strategy="fixed",
                chunk_tokens=chunk_tokens,
                overlap_tokens=8,
                tokenizer="BAAI/bge-base-en-v1.5",
            ),
        ),
        index=IndexConfig(
            backend=backend,
            embedding=EmbeddingConfig(
                model=model,
                normalize=normalize,
                batch_size=4,
                device="cpu",
                query_instruction=query_instruction,
            ),
            faiss=FaissConfig(index_type="flat_ip"),
            bm25=BM25Config(enabled=False, k1=1.5, b=0.75),
        ),
        retrieve=RetrieveConfig(
            mode=mode,
            top_k=top_k,
            candidate_k=50,
            fusion=FusionConfig(method="rrf", rrf_k=60),
            rerank=RerankConfig(enabled=False, model="m", top_n=10),
        ),
        generate=GenerateConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=100,
            effort="medium",
            cite_chunks=True,
            abstain_if_no_evidence=True,
        ),
        eval=EvalConfig(
            golden_set=tmp_path / "golden.jsonl",
            results_dir=tmp_path / "results",
            k_values=(1, 3, 5),
            ragas=RagasConfig(enabled=False, metrics=()),
            seed=0,
        ),
    )


class FakeEmbedder:
    """Hash-based embeddings: deterministic, no model download, no torch.

    Every distinct text gets a stable pseudo-random unit vector, so identical
    text embeds identically (which is what the round-trip and pairing tests
    need) while different text is near-orthogonal.
    """

    def __init__(self, config: EmbeddingConfig, dim: int = DIM):
        self.config = config
        self._dim = dim
        self.device = "cpu"
        self.metric = "cosine" if config.normalize else "dot"
        self.encoded_passages: list[str] = []
        self.encoded_queries: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return 512

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).normal(size=self._dim)
        if self.config.normalize:
            vector /= np.linalg.norm(vector)
        return vector

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(
            np.vstack([self._vector(t) for t in texts]), dtype=np.float32
        )

    def encode_passages(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        self.encoded_passages.extend(texts)
        return self._encode(texts)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefix = self.config.query_instruction
        prepared = [prefix + q for q in queries] if prefix else list(queries)
        self.encoded_queries.extend(prepared)
        return self._encode(prepared)

    def truncation_report(self, texts: list[str]):
        from quant_rag.index.embed import TruncationReport

        return TruncationReport(len(texts), 0, 0, self.max_seq_length)


def write_chunks(config: PipelineConfig, bodies: list[tuple[str, str]]) -> list[Chunk]:
    """Write a chunk set of ``(doc_id, text)`` pairs where the config expects it."""
    chunks: list[Chunk] = []
    per_doc: dict[str, int] = {}
    for doc_id, text in bodies:
        index = per_doc.get(doc_id, 0)
        per_doc[doc_id] = index + 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{index:04d}",
                doc_id=doc_id,
                index=index,
                text=text,
                char_start=0,
                char_end=len(text),
                n_tokens=len(text.split()),
                strategy="fixed",
                section_title="S",
            )
        )
    path = store.chunks_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return chunks


@pytest.fixture
def built(tmp_path):
    """A config with a chunk set and a built index, using FakeEmbedder."""
    config = make_config(tmp_path)
    chunks = write_chunks(
        config,
        [
            ("arxiv_1706.10059", "the EIIE topology is an ensemble of identical evaluators"),
            ("arxiv_1706.10059", "portfolio vector memory avoids gradient vanishing"),
            ("note_afml_labeling", "triple-barrier labeling sets three barriers per trade"),
            ("note_afml_labeling", "purging and embargo remove leaked observations"),
        ],
    )
    embedder = FakeEmbedder(config.index.embedding)
    index, directory = build_index(config, embedder=embedder, log=lambda _m: None)
    return config, chunks, index, directory, embedder


# --------------------------------------------------------------------------- #
# Fingerprint pairing
# --------------------------------------------------------------------------- #


def test_index_fingerprint_depends_on_the_ingest_fingerprint(tmp_path):
    """Changing a chunking knob must move the index, not just the chunk set."""
    base = make_config(tmp_path, chunk_tokens=64)
    other = make_config(tmp_path, chunk_tokens=128)
    assert base.ingest_fingerprint() != other.ingest_fingerprint()
    assert base.index_fingerprint() != other.index_fingerprint()
    assert index_dir(base) != index_dir(other)


def test_index_fingerprint_depends_on_the_embedding_knobs(tmp_path):
    base = make_config(tmp_path)
    assert base.index_fingerprint() != make_config(tmp_path, model="other").index_fingerprint()
    assert (
        base.index_fingerprint() != make_config(tmp_path, normalize=False).index_fingerprint()
    )


def test_index_fingerprint_ignores_query_side_and_perf_knobs(tmp_path):
    """query_instruction changes no stored vector, so it must not force a rebuild."""
    base = make_config(tmp_path, query_instruction="Q: ")
    assert make_config(tmp_path, query_instruction="").index_fingerprint() == (
        base.index_fingerprint()
    )


def test_index_fingerprint_ignores_the_backend(tmp_path):
    """Both backends are exact searches over the same vectors, so they share one
    index directory and switching never costs a re-embed."""
    base = make_config(tmp_path, backend="exact_numpy")
    other = make_config(tmp_path, backend="faiss_flat_ip")
    assert base.index_fingerprint() == other.index_fingerprint()
    assert index_dir(base) == index_dir(other)


def test_index_fingerprint_is_stable_across_calls(tmp_path):
    config = make_config(tmp_path)
    assert config.index_fingerprint() == config.index_fingerprint()
    assert len(config.index_fingerprint()) == 10


def test_load_rejects_an_index_whose_chunkset_changed_underneath(built):
    """The guarantee that matters: same fingerprint, different chunk bytes.

    Re-ingesting with an exact tokenizer where the fallback was used before
    leaves the ingest fingerprint untouched but moves every chunk boundary.
    Only the recorded content hash catches that.
    """
    config, _chunks, _index, _directory, _embedder = built
    load_index(config)  # verifies before the edit

    path = store.chunks_path(config)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["text"] = record["text"] + " and one more sentence"
    lines[0] = json.dumps(record, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert config.index_fingerprint() == _index.meta.index_fingerprint
    with pytest.raises(IndexPairingError, match="chunk set changed"):
        load_index(config)


def test_load_rejects_a_chunkset_with_a_different_number_of_chunks(built):
    config, _chunks, _index, _directory, _embedder = built
    path = store.chunks_path(config)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(IndexPairingError):
        load_index(config)


def test_load_rejects_reordered_chunk_ids(built):
    """Row i of the index means chunk i; order is part of the pairing."""
    config, _chunks, _index, directory, _embedder = built
    ids_file = directory / store.CHUNK_IDS_FILE
    stored = json.loads(ids_file.read_text(encoding="utf-8"))
    ids_file.write_text(json.dumps(list(reversed(stored))), encoding="utf-8")
    with pytest.raises(IndexPairingError, match="chunk-id order"):
        load_index(config)


def test_load_rejects_metadata_from_a_foreign_fingerprint(built):
    config, _chunks, _index, directory, _embedder = built
    meta_file = directory / store.META_FILE
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["index_fingerprint"] = "deadbeef00"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(IndexPairingError, match="index_fingerprint"):
        load_index(config)


def test_load_rejects_a_foreign_ingest_fingerprint(built):
    config, _chunks, _index, directory, _embedder = built
    meta_file = directory / store.META_FILE
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["ingest_fingerprint"] = "deadbeef00"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(IndexPairingError, match="ingest fingerprint"):
        load_index(config)


def test_load_without_an_index_is_a_missing_error_not_a_pairing_error(tmp_path):
    config = make_config(tmp_path)
    write_chunks(config, [("d", "some text")])
    with pytest.raises(IndexMissingError, match="no index for fingerprint"):
        load_index(config)


def test_build_without_a_chunkset_says_to_ingest_first(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(IndexMissingError, match="quant-rag ingest"):
        build_index(config, embedder=FakeEmbedder(config.index.embedding), log=lambda _m: None)


def test_duplicate_chunk_ids_are_rejected(tmp_path):
    config = make_config(tmp_path)
    path = store.chunks_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = Chunk(
        chunk_id="d#0000",
        doc_id="d",
        index=0,
        text="t",
        char_start=0,
        char_end=1,
        n_tokens=1,
        strategy="fixed",
    )
    line = json.dumps(chunk.to_dict())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(IndexPairingError, match="duplicate chunk_id"):
        store.load_chunks(path)


# --------------------------------------------------------------------------- #
# Round-trip persistence
# --------------------------------------------------------------------------- #


def test_index_round_trips_through_disk(built):
    config, chunks, index, directory, _embedder = built
    assert (directory / store.VECTORS_FILE).exists()
    assert (directory / store.CHUNK_IDS_FILE).exists()
    assert (directory / store.META_FILE).exists()

    reloaded = load_index(config)
    assert reloaded.meta == index.meta
    assert reloaded.chunk_ids == [c.chunk_id for c in chunks]
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in chunks]
    assert [c.text for c in reloaded.chunks] == [c.text for c in chunks]
    assert reloaded.backend.n_vectors == len(chunks)
    assert reloaded.backend.name == "exact_numpy"
    assert reloaded.vectors.shape == (len(chunks), DIM)
    assert reloaded.meta.dim == DIM
    assert reloaded.meta.metric == "cosine"
    assert reloaded.meta.n_vectors == len(chunks)


def test_reloaded_index_returns_identical_scores(built):
    """Persistence must not perturb the vectors; same query, same ranking."""
    config, _chunks, index, _directory, embedder = built
    query_vector = embedder.encode_queries(["triple-barrier labeling"])

    before_scores, before_rows = index.search(query_vector, 4)
    reloaded = load_index(config)
    after_scores, after_rows = reloaded.search(query_vector, 4)

    assert (before_rows == after_rows).all()
    assert before_scores == pytest.approx(after_scores)
    assert [index.chunk_at(r).chunk_id for r in before_rows[0]] == [
        reloaded.chunk_at(r).chunk_id for r in after_rows[0]
    ]
    # Byte-identical vectors, not merely equivalent rankings.
    assert np.array_equal(index.vectors, reloaded.vectors)


def test_metadata_records_what_it_was_built_from(built):
    config, chunks, index, _directory, _embedder = built
    meta = index.meta
    assert meta.ingest_fingerprint == config.ingest_fingerprint()
    assert meta.index_fingerprint == config.index_fingerprint()
    assert meta.chunks_sha256 == store.file_digest(store.chunks_path(config))
    assert meta.model == config.index.embedding.model
    assert meta.normalize is True
    assert meta.backend == "exact_numpy"
    assert meta.n_vectors == len(chunks)
    assert meta.truncation["model_max_seq_length"] == 512


def test_vectors_are_added_in_chunkset_order(built):
    """Row i of the FAISS index must be chunk i, or every chunk_id is wrong."""
    config, chunks, index, _directory, embedder = built
    reloaded = load_index(config)
    for row, chunk in enumerate(chunks):
        vector = embedder.encode_passages([chunk.text])
        scores, rows = reloaded.search(vector, 1)
        assert rows[0][0] == row
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)
        assert reloaded.chunk_at(row).chunk_id == chunk.chunk_id


def test_meta_from_dict_rejects_truncated_metadata(built):
    _config, _chunks, index, directory, _embedder = built
    meta_file = directory / store.META_FILE
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    del meta["chunks_sha256"]
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(IndexPairingError, match="missing field"):
        load_index(_config)


# --------------------------------------------------------------------------- #
# Config validation added with this stage
# --------------------------------------------------------------------------- #


def test_unsupported_index_type_is_a_config_error():
    with pytest.raises(ConfigError, match="index_type"):
        FaissConfig.parse({"index_type": "hnsw"})


def test_unknown_device_is_a_config_error():
    with pytest.raises(ConfigError, match="device"):
        EmbeddingConfig.parse(
            {
                "model": "m",
                "normalize": True,
                "batch_size": 8,
                "device": "tpu",
                "query_instruction": "",
            }
        )


def test_query_instruction_is_required_but_may_be_empty():
    with pytest.raises(ConfigError, match="query_instruction"):
        EmbeddingConfig.parse(
            {"model": "m", "normalize": True, "batch_size": 8, "device": "cpu"}
        )
    cfg = EmbeddingConfig.parse(
        {
            "model": "m",
            "normalize": True,
            "batch_size": 8,
            "device": "cpu",
            "query_instruction": "",
        }
    )
    assert cfg.query_instruction == ""


def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_special_token_budget_is_two():
    """The constant the truncation warning is computed from."""
    assert N_SPECIAL_TOKENS == 2


# --------------------------------------------------------------------------- #
# The duplicate-libomp workaround in index._runtime
# --------------------------------------------------------------------------- #


def test_both_openmp_runtimes_are_pinned():
    """``_runtime`` is only safe because *neither* library builds a thread pool.

    Pinning faiss alone was not enough during development -- a multithreaded
    torch matmul then segfaulted the process -- so both pins are part of the
    contract and both are asserted here.
    """
    from quant_rag.index._runtime import faiss, torch  # noqa: PLC0415

    assert faiss.omp_get_max_threads() == 1
    assert torch.get_num_threads() == 1
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_faiss_search_matches_exact_numpy_search():
    """Guard for the ``KMP_DUPLICATE_LIB_OK`` workaround in ``index._runtime``.

    That workaround is documented as able to "silently produce incorrect
    results". This asserts, in a process where torch's OpenMP runtime is also
    live, that flat_ip search returns exactly what a brute-force ``Q @ V.T``
    returns -- same rows, same order, same scores to float32 precision. If the
    duplicate runtime ever does corrupt a result, this stops being silent.
    """
    from quant_rag.index._runtime import torch  # noqa: PLC0415

    # Exercise torch so its runtime is really initialized, not merely imported.
    assert float((torch.randn(300, 300) @ torch.randn(300, 300)).sum()) is not None

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(461, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(8, 768)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    from quant_rag.index.backends import FaissFlatIPBackend  # noqa: PLC0415

    index = FaissFlatIPBackend(np.ascontiguousarray(vectors))
    scores, rows = index.search(np.ascontiguousarray(queries), 10)

    reference = queries @ vectors.T
    expected_rows = np.argsort(-reference, axis=1)[:, :10]
    assert (rows == expected_rows).all()
    assert scores == pytest.approx(np.take_along_axis(reference, rows, axis=1), abs=1e-5)
