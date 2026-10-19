"""Build, persist, and load the vector index — and refuse mismatched pairings.

Layout under ``corpus.processed_dir`` (gitignored; CLAUDE.md rule 2):

    index/<index_fingerprint>/vectors.npy     (n_chunks, dim) float32
    index/<index_fingerprint>/chunk_ids.json  row i of the matrix -> chunk_id
    index/<index_fingerprint>/index.json      what it was built from

The persisted artifact is the vector matrix, not a search structure. Both
backends (:mod:`quant_rag.index.backends`) are exact searches constructed from
that matrix at load time, so the stored bytes do not depend on which one is
configured and switching backends never costs a re-embed.

An index is meaningless without the exact chunk set it was built from: row 37
means "chunk 37 of *that* chunks.jsonl". Pairing is enforced at two levels:

1. **Path.** ``index_fingerprint`` hashes the ingest fingerprint along with the
   embedding knobs, so a different chunking config gets a different directory
   and can never overwrite or masquerade as this one.
2. **Content.** ``index.json`` records the SHA-256 of the chunks.jsonl bytes
   and the chunk-id list. Loading re-hashes the file and raises
   :class:`IndexPairingError` on any difference. This is the level that
   matters, because the fingerprint hashes the tokenizer *name*: re-ingesting
   after ``transformers`` became importable produces different chunk
   boundaries under an unchanged fingerprint, and only the content hash
   notices.

Loading never silently rebuilds or repairs. A stale index is an error with an
instruction, so no eval run can report numbers from vectors that do not match
the chunks whose IDs it is scoring.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quant_rag.config import PipelineConfig
from quant_rag.index.backends import SearchBackend, build_backend
from quant_rag.ingest.chunk import Chunk

VECTORS_FILE = "vectors.npy"
CHUNK_IDS_FILE = "chunk_ids.json"
META_FILE = "index.json"


class IndexStoreError(Exception):
    """Base class for index problems."""


class IndexMissingError(IndexStoreError):
    """No index exists for this config yet."""


class IndexPairingError(IndexStoreError):
    """The index on disk was not built from the chunk set it is paired with."""


# --------------------------------------------------------------------------- #
# Chunk sets
# --------------------------------------------------------------------------- #


def chunks_path(config: PipelineConfig) -> Path:
    processed = config.resolve(config.corpus.processed_dir)
    return processed / "chunks" / config.ingest_fingerprint() / "chunks.jsonl"


def index_dir(config: PipelineConfig) -> Path:
    processed = config.resolve(config.corpus.processed_dir)
    return processed / "index" / config.index_fingerprint()


def file_digest(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_chunks(path: Path) -> list[Chunk]:
    """Read ``chunks.jsonl`` in file order. Row order is index row order."""
    if not path.exists():
        raise IndexMissingError(
            f"chunk set not found: {path}\nrun `uv run quant-rag ingest` first"
        )
    with path.open(encoding="utf-8") as fh:
        chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
    if not chunks:
        raise IndexMissingError(f"chunk set is empty: {path}")
    duplicates = len(chunks) - len({c.chunk_id for c in chunks})
    if duplicates:
        # chunk_id is the eval's join key; duplicates would make recall@k
        # ambiguous, so this is a build-time error rather than a warning.
        raise IndexPairingError(f"{path}: {duplicates} duplicate chunk_id(s)")
    return chunks


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexMeta:
    """Everything needed to prove an index matches a chunk set and a model.

    ``backend`` records which backend *built* it, for the record only -- the
    stored vectors are backend-neutral, so loading under the other backend is
    allowed and is how the two get compared.
    """

    index_fingerprint: str
    ingest_fingerprint: str
    experiment: str
    chunks_file: str
    chunks_sha256: str
    n_vectors: int
    dim: int
    model: str
    normalize: bool
    metric: str
    backend: str
    device: str
    built_at: str
    truncation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMeta:
        known = set(cls.__dataclass_fields__)
        missing = known - set(data)
        if missing:
            raise IndexPairingError(
                f"{META_FILE} is missing field(s) {sorted(missing)}; "
                "it was written by an older build -- rebuild with "
                "`quant-rag index --rebuild`"
            )
        return cls(**{k: data[k] for k in known})


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save_index(
    directory: Path, vectors: np.ndarray, chunk_ids: list[str], meta: IndexMeta
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTORS_FILE, np.ascontiguousarray(vectors, dtype=np.float32))
    (directory / CHUNK_IDS_FILE).write_text(
        json.dumps(chunk_ids, indent=2) + "\n", encoding="utf-8"
    )
    (directory / META_FILE).write_text(
        json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


@dataclass
class LoadedIndex:
    """A search backend with the chunks it was built from, verified as a pair."""

    backend: SearchBackend
    vectors: np.ndarray
    chunks: list[Chunk]
    chunk_ids: list[str]
    meta: IndexMeta

    def chunk_at(self, row: int) -> Chunk:
        return self.chunks[row]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.backend.search(queries, k)


def load_index(config: PipelineConfig) -> LoadedIndex:
    """Load the index for ``config``, verifying it against its chunk set.

    Raises :class:`IndexMissingError` if nothing is built yet, and
    :class:`IndexPairingError` if what is on disk does not match the chunk set
    the config currently points at.
    """
    directory = index_dir(config)
    expected_fingerprint = config.index_fingerprint()
    if not (directory / META_FILE).exists():
        raise IndexMissingError(
            f"no index for fingerprint {expected_fingerprint} at {directory}\n"
            "run `uv run quant-rag index` first"
        )

    meta = IndexMeta.from_dict(
        json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    )
    # The directory name is derived, so a mismatch here means the metadata was
    # moved or hand-edited; distrust it rather than the config.
    if meta.index_fingerprint != expected_fingerprint:
        raise IndexPairingError(
            f"{directory / META_FILE}: index_fingerprint {meta.index_fingerprint!r} "
            f"does not match this config's {expected_fingerprint!r}"
        )
    if meta.ingest_fingerprint != config.ingest_fingerprint():
        raise IndexPairingError(
            f"index was built from ingest fingerprint {meta.ingest_fingerprint!r}, "
            f"config now specifies {config.ingest_fingerprint()!r}"
        )

    path = chunks_path(config)
    chunks = load_chunks(path)
    digest = file_digest(path)
    if digest != meta.chunks_sha256:
        raise IndexPairingError(
            f"chunk set changed since the index was built\n"
            f"  index.json records sha256 {meta.chunks_sha256[:16]}...\n"
            f"  {path} is now sha256 {digest[:16]}...\n"
            "The vectors no longer correspond to these chunks. Rebuild with "
            "`uv run quant-rag index --rebuild`."
        )

    chunk_ids = json.loads((directory / CHUNK_IDS_FILE).read_text(encoding="utf-8"))
    if chunk_ids != [c.chunk_id for c in chunks]:
        raise IndexPairingError(
            f"{directory / CHUNK_IDS_FILE}: stored chunk-id order does not match "
            f"{path}; rebuild with `uv run quant-rag index --rebuild`"
        )

    vectors = np.load(directory / VECTORS_FILE, allow_pickle=False)
    if vectors.shape[0] != len(chunks):
        raise IndexPairingError(
            f"index holds {vectors.shape[0]} vectors but the chunk set has "
            f"{len(chunks)} chunks; rebuild with `uv run quant-rag index --rebuild`"
        )
    if vectors.shape[1] != meta.dim:
        raise IndexPairingError(
            f"{VECTORS_FILE} has dim {vectors.shape[1]} but index.json records "
            f"{meta.dim}; rebuild with `uv run quant-rag index --rebuild`"
        )

    backend = build_backend(
        vectors, config.index.backend, index_type=config.index.faiss.index_type
    )
    return LoadedIndex(
        backend=backend,
        vectors=vectors,
        chunks=chunks,
        chunk_ids=chunk_ids,
        meta=meta,
    )


def build_index(
    config: PipelineConfig,
    *,
    embedder: Any = None,
    log: Callable[[str], None] = print,
) -> tuple[LoadedIndex, Path]:
    """Embed the active chunk set and write the vectors to disk."""
    from quant_rag.index.embed import Embedder  # noqa: PLC0415 - avoids a cycle

    path = chunks_path(config)
    chunks = load_chunks(path)
    digest = file_digest(path)
    embedding = config.index.embedding

    if embedder is None:
        embedder = Embedder(embedding)
    log(f"chunk set: {path} ({len(chunks)} chunks, sha256 {digest[:12]}...)")
    log(
        f"model: {embedding.model} dim={embedder.dim} device={embedder.device} "
        f"normalize={embedding.normalize} batch_size={embedding.batch_size}"
    )
    log(f"backend: {config.index.backend}")

    texts = [c.text for c in chunks]
    truncation = embedder.truncation_report(texts)
    log(str(truncation))

    vectors = embedder.encode_passages(texts, show_progress=True)
    if vectors.shape[0] != len(chunks):  # pragma: no cover - defensive
        raise RuntimeError(f"encoded {vectors.shape[0]} vectors for {len(chunks)} chunks")
    if embedding.normalize:
        # Inner product only equals cosine if the vectors really are unit
        # length; assert it instead of trusting the flag through two libraries.
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError(
                f"normalize=true but vector norms span "
                f"[{norms.min():.4f}, {norms.max():.4f}]"
            )

    meta = IndexMeta(
        index_fingerprint=config.index_fingerprint(),
        ingest_fingerprint=config.ingest_fingerprint(),
        experiment=config.experiment.name,
        chunks_file=str(path),
        chunks_sha256=digest,
        n_vectors=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        model=embedding.model,
        normalize=embedding.normalize,
        metric=embedder.metric,
        backend=config.index.backend,
        device=embedder.device,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        truncation={
            "n_texts": truncation.n_texts,
            "n_truncated": truncation.n_truncated,
            "max_tokens_dropped": truncation.max_tokens_dropped,
            "model_max_seq_length": truncation.limit,
        },
    )
    directory = index_dir(config)
    save_index(directory, vectors, [c.chunk_id for c in chunks], meta)

    backend = build_backend(
        vectors, config.index.backend, index_type=config.index.faiss.index_type
    )
    return (
        LoadedIndex(
            backend=backend,
            vectors=vectors,
            chunks=chunks,
            chunk_ids=[c.chunk_id for c in chunks],
            meta=meta,
        ),
        directory,
    )
