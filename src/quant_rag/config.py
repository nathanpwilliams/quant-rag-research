"""Typed loader for ``config/pipeline.yaml``.

Every experiment knob lives in a config file (CLAUDE.md rule 3), so this module
is the single place that turns YAML into typed objects. Two properties matter
for the ablation study:

* **No defaults that silently differ from the file.** Every knob below is
  required; a config missing ``chunk_tokens`` is an error, not a fallback to
  512. Otherwise a results CSV could record a chunk size the file never set.
* **Unknown keys are errors.** A typo like ``chunk_token: 512`` in a new
  ablation YAML would otherwise be ignored and silently run the baseline.

``fingerprint()`` gives a short hash of the ingest-relevant knobs; chunk sets
are written under that fingerprint so two chunking configs can coexist on disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

CHUNK_STRATEGIES = frozenset({"fixed", "structure"})
PARSERS = frozenset({"pymupdf", "pymupdf4llm"})
RETRIEVE_MODES = frozenset({"dense", "sparse", "hybrid"})
INDEX_TYPES = frozenset({"flat_ip"})
INDEX_BACKENDS = frozenset({"exact_numpy", "faiss_flat_ip"})
DEVICES = frozenset({"auto", "cpu", "mps", "cuda"})


class ConfigError(ValueError):
    """The config file is missing a knob, has an unknown key, or a bad value."""


def _section(data: dict[str, Any], key: str, *, path: str = "") -> dict[str, Any]:
    where = f"{path}.{key}" if path else key
    if key not in data:
        raise ConfigError(f"missing config section: {where}")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _take(data: dict[str, Any], key: str, kind: type | tuple[type, ...], path: str) -> Any:
    if key not in data:
        raise ConfigError(f"missing config key: {path}.{key}")
    value = data[key]
    # bool is an int subclass; an int knob must not accept `true`.
    if kind is not bool and isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected {kind}, got bool")
    if not isinstance(value, kind):
        raise ConfigError(f"{path}.{key}: expected {kind}, got {type(value).__name__}")
    return value


def _short_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def _reject_unknown(data: dict[str, Any], known: tuple[str, ...], path: str) -> None:
    extra = sorted(set(data) - set(known))
    if extra:
        raise ConfigError(f"{path}: unknown key(s) {extra}; known keys are {sorted(known)}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str

    @classmethod
    def parse(cls, data: dict[str, Any]) -> ExperimentConfig:
        _reject_unknown(data, ("name", "description"), "experiment")
        return cls(
            name=_take(data, "name", str, "experiment"),
            description=_take(data, "description", str, "experiment"),
        )


@dataclass(frozen=True)
class CorpusConfig:
    manifest: Path
    raw_dir: Path
    processed_dir: Path

    @classmethod
    def parse(cls, data: dict[str, Any]) -> CorpusConfig:
        keys = ("manifest", "raw_dir", "processed_dir")
        _reject_unknown(data, keys, "corpus")
        return cls(**{k: Path(_take(data, k, str, "corpus")) for k in keys})


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    chunk_tokens: int
    overlap_tokens: int
    tokenizer: str

    @classmethod
    def parse(cls, data: dict[str, Any]) -> ChunkingConfig:
        path = "ingest.chunking"
        _reject_unknown(data, ("strategy", "chunk_tokens", "overlap_tokens", "tokenizer"), path)
        strategy = _take(data, "strategy", str, path)
        if strategy not in CHUNK_STRATEGIES:
            raise ConfigError(
                f"{path}.strategy: {strategy!r} not in {sorted(CHUNK_STRATEGIES)}"
            )
        chunk_tokens = _take(data, "chunk_tokens", int, path)
        overlap_tokens = _take(data, "overlap_tokens", int, path)
        if chunk_tokens <= 0:
            raise ConfigError(f"{path}.chunk_tokens: must be positive, got {chunk_tokens}")
        if overlap_tokens < 0:
            raise ConfigError(f"{path}.overlap_tokens: must be >= 0, got {overlap_tokens}")
        if overlap_tokens >= chunk_tokens:
            # Equal would make the window never advance; the loop would not terminate.
            raise ConfigError(
                f"{path}: overlap_tokens ({overlap_tokens}) must be < "
                f"chunk_tokens ({chunk_tokens})"
            )
        return cls(
            strategy=strategy,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer=_take(data, "tokenizer", str, path),
        )


@dataclass(frozen=True)
class IngestConfig:
    parser: str
    chunking: ChunkingConfig

    @classmethod
    def parse(cls, data: dict[str, Any]) -> IngestConfig:
        _reject_unknown(data, ("parser", "chunking"), "ingest")
        parser = _take(data, "parser", str, "ingest")
        if parser not in PARSERS:
            raise ConfigError(f"ingest.parser: {parser!r} not in {sorted(PARSERS)}")
        return cls(parser=parser, chunking=ChunkingConfig.parse(_section(data, "chunking", path="ingest")))


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding knobs.

    ``query_instruction`` is prepended to *queries only*, never to passages --
    that asymmetry is what the BGE model card prescribes. Because passage
    vectors are unaffected by it, it is deliberately NOT part of
    ``index_fingerprint()``: flipping it is a retrieval-side ablation that
    reuses the same index rather than forcing a rebuild.
    """

    model: str
    normalize: bool
    batch_size: int
    device: str
    query_instruction: str

    @classmethod
    def parse(cls, data: dict[str, Any]) -> EmbeddingConfig:
        path = "index.embedding"
        keys = ("model", "normalize", "batch_size", "device", "query_instruction")
        _reject_unknown(data, keys, path)
        batch_size = _take(data, "batch_size", int, path)
        if batch_size <= 0:
            raise ConfigError(f"{path}.batch_size: must be positive, got {batch_size}")
        device = _take(data, "device", str, path)
        if device not in DEVICES:
            raise ConfigError(f"{path}.device: {device!r} not in {sorted(DEVICES)}")
        return cls(
            model=_take(data, "model", str, path),
            normalize=_take(data, "normalize", bool, path),
            batch_size=batch_size,
            device=device,
            # Empty string means "no instruction", which is a valid setting for
            # bge-*-v1.5 (see config/pipeline.yaml), so this knob is not
            # required to be non-empty -- only to be present.
            query_instruction=_take(data, "query_instruction", str, path),
        )


@dataclass(frozen=True)
class FaissConfig:
    index_type: str

    @classmethod
    def parse(cls, data: dict[str, Any]) -> FaissConfig:
        _reject_unknown(data, ("index_type",), "index.faiss")
        index_type = _take(data, "index_type", str, "index.faiss")
        if index_type not in INDEX_TYPES:
            raise ConfigError(
                f"index.faiss.index_type: {index_type!r} not in {sorted(INDEX_TYPES)}"
            )
        return cls(index_type=index_type)


@dataclass(frozen=True)
class BM25Config:
    enabled: bool
    k1: float
    b: float

    @classmethod
    def parse(cls, data: dict[str, Any]) -> BM25Config:
        path = "index.bm25"
        _reject_unknown(data, ("enabled", "k1", "b"), path)
        return cls(
            enabled=_take(data, "enabled", bool, path),
            k1=float(_take(data, "k1", (int, float), path)),
            b=float(_take(data, "b", (int, float), path)),
        )


@dataclass(frozen=True)
class IndexConfig:
    """Index knobs.

    ``backend`` selects the *search* structure over the stored vectors, not the
    vectors themselves -- both backends are exact, and both read the same
    persisted ``vectors.npy``. It is therefore not part of
    ``index_fingerprint()``: switching backends re-uses the existing
    embeddings instead of forcing a re-embed.
    """

    backend: str
    embedding: EmbeddingConfig
    faiss: FaissConfig
    bm25: BM25Config

    @classmethod
    def parse(cls, data: dict[str, Any]) -> IndexConfig:
        _reject_unknown(data, ("backend", "embedding", "faiss", "bm25"), "index")
        backend = _take(data, "backend", str, "index")
        if backend not in INDEX_BACKENDS:
            raise ConfigError(f"index.backend: {backend!r} not in {sorted(INDEX_BACKENDS)}")
        return cls(
            backend=backend,
            embedding=EmbeddingConfig.parse(_section(data, "embedding", path="index")),
            faiss=FaissConfig.parse(_section(data, "faiss", path="index")),
            bm25=BM25Config.parse(_section(data, "bm25", path="index")),
        )


@dataclass(frozen=True)
class FusionConfig:
    method: str
    rrf_k: int

    @classmethod
    def parse(cls, data: dict[str, Any]) -> FusionConfig:
        path = "retrieve.fusion"
        _reject_unknown(data, ("method", "rrf_k"), path)
        return cls(
            method=_take(data, "method", str, path),
            rrf_k=_take(data, "rrf_k", int, path),
        )


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool
    model: str
    top_n: int

    @classmethod
    def parse(cls, data: dict[str, Any]) -> RerankConfig:
        path = "retrieve.rerank"
        _reject_unknown(data, ("enabled", "model", "top_n"), path)
        return cls(
            enabled=_take(data, "enabled", bool, path),
            model=_take(data, "model", str, path),
            top_n=_take(data, "top_n", int, path),
        )


@dataclass(frozen=True)
class RetrieveConfig:
    mode: str
    top_k: int
    candidate_k: int
    fusion: FusionConfig
    rerank: RerankConfig

    @classmethod
    def parse(cls, data: dict[str, Any]) -> RetrieveConfig:
        path = "retrieve"
        _reject_unknown(data, ("mode", "top_k", "candidate_k", "fusion", "rerank"), path)
        mode = _take(data, "mode", str, path)
        if mode not in RETRIEVE_MODES:
            raise ConfigError(f"{path}.mode: {mode!r} not in {sorted(RETRIEVE_MODES)}")
        return cls(
            mode=mode,
            top_k=_take(data, "top_k", int, path),
            candidate_k=_take(data, "candidate_k", int, path),
            fusion=FusionConfig.parse(_section(data, "fusion", path=path)),
            rerank=RerankConfig.parse(_section(data, "rerank", path=path)),
        )


@dataclass(frozen=True)
class GenerateConfig:
    provider: str
    model: str
    max_tokens: int
    effort: str
    cite_chunks: bool
    abstain_if_no_evidence: bool

    @classmethod
    def parse(cls, data: dict[str, Any]) -> GenerateConfig:
        path = "generate"
        keys = (
            "provider",
            "model",
            "max_tokens",
            "effort",
            "cite_chunks",
            "abstain_if_no_evidence",
        )
        _reject_unknown(data, keys, path)
        return cls(
            provider=_take(data, "provider", str, path),
            model=_take(data, "model", str, path),
            max_tokens=_take(data, "max_tokens", int, path),
            effort=_take(data, "effort", str, path),
            cite_chunks=_take(data, "cite_chunks", bool, path),
            abstain_if_no_evidence=_take(data, "abstain_if_no_evidence", bool, path),
        )


@dataclass(frozen=True)
class RagasConfig:
    enabled: bool
    metrics: tuple[str, ...]

    @classmethod
    def parse(cls, data: dict[str, Any]) -> RagasConfig:
        path = "eval.ragas"
        _reject_unknown(data, ("enabled", "metrics"), path)
        return cls(
            enabled=_take(data, "enabled", bool, path),
            metrics=tuple(_take(data, "metrics", list, path)),
        )


@dataclass(frozen=True)
class EvalConfig:
    golden_set: Path
    results_dir: Path
    k_values: tuple[int, ...]
    ragas: RagasConfig
    seed: int

    @classmethod
    def parse(cls, data: dict[str, Any]) -> EvalConfig:
        path = "eval"
        _reject_unknown(data, ("golden_set", "results_dir", "k_values", "ragas", "seed"), path)
        return cls(
            golden_set=Path(_take(data, "golden_set", str, path)),
            results_dir=Path(_take(data, "results_dir", str, path)),
            k_values=tuple(_take(data, "k_values", list, path)),
            ragas=RagasConfig.parse(_section(data, "ragas", path=path)),
            seed=_take(data, "seed", int, path),
        )


@dataclass(frozen=True)
class PipelineConfig:
    experiment: ExperimentConfig
    corpus: CorpusConfig
    ingest: IngestConfig
    index: IndexConfig
    retrieve: RetrieveConfig
    generate: GenerateConfig
    eval: EvalConfig
    source_path: Path | None = None

    @classmethod
    def parse(cls, data: dict[str, Any], *, source_path: Path | None = None) -> PipelineConfig:
        if not isinstance(data, dict):
            raise ConfigError(f"config root: expected a mapping, got {type(data).__name__}")
        sections = ("experiment", "corpus", "ingest", "index", "retrieve", "generate", "eval")
        _reject_unknown(data, sections, "config root")
        return cls(
            experiment=ExperimentConfig.parse(_section(data, "experiment")),
            corpus=CorpusConfig.parse(_section(data, "corpus")),
            ingest=IngestConfig.parse(_section(data, "ingest")),
            index=IndexConfig.parse(_section(data, "index")),
            retrieve=RetrieveConfig.parse(_section(data, "retrieve")),
            generate=GenerateConfig.parse(_section(data, "generate")),
            eval=EvalConfig.parse(_section(data, "eval")),
            source_path=source_path,
        )

    # Paths in the config are repo-relative so a config file is portable.
    def resolve(self, path: Path, *, root: Path = REPO_ROOT) -> Path:
        return path if path.is_absolute() else root / path

    def ingest_fingerprint(self) -> str:
        """Short hash of the knobs that determine chunk content and boundaries.

        Chunk sets are stored under this fingerprint, so switching ablation rows
        does not overwrite a previous row's chunks (CLAUDE.md rule 4 in spirit).
        """
        payload = {
            "parser": self.ingest.parser,
            "chunking": asdict(self.ingest.chunking),
        }
        return _short_hash(payload)

    def index_fingerprint(self) -> str:
        """Short hash of the knobs that determine the stored vectors.

        The ingest fingerprint is an *input* to this hash, so an index
        directory is structurally bound to one chunking config: changing a
        chunking knob changes the index fingerprint too, and the old index
        cannot be mistaken for a current one. That is necessary but not
        sufficient -- two chunk sets can share an ingest fingerprint (the
        tokenizer *name* is hashed, not whether it resolved to the exact HF
        tokenizer or the regex fallback), so the stored index metadata also
        pins the chunk file's content hash. See ``index.store``.

        Excluded on purpose: ``batch_size`` and ``device`` (they do not change
        what a vector means), ``query_instruction`` (query-side only), and
        ``backend``/``faiss.index_type`` (both backends are exact searches over
        the same vectors, so they share one set of embeddings).
        """
        payload = {
            "ingest": self.ingest_fingerprint(),
            "model": self.index.embedding.model,
            "normalize": self.index.embedding.normalize,
        }
        return _short_hash(payload)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    try:
        return PipelineConfig.parse(data, source_path=path)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
