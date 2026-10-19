"""Stage wiring: ingest -> index -> retrieve -> generate.

Each stage reads its knobs from the ``config/*.yaml`` file passed on the CLI;
an ablation row is a config file, never a code branch. Stage implementations
land in their own packages (``ingest/``, ``index/``, ``retrieve/``,
``generate/``); this module only composes them.

Stage status:

* ``ingest`` -- implemented: :func:`quant_rag.ingest.run.ingest_corpus` parses
  the manifest's documents to ``data/processed/parsed/`` and writes a chunk set
  keyed by :meth:`quant_rag.config.PipelineConfig.ingest_fingerprint`.
* ``index`` -- implemented: :func:`quant_rag.index.store.build_index` embeds
  that chunk set and writes a FAISS index keyed by
  :meth:`quant_rag.config.PipelineConfig.index_fingerprint`, which has the
  ingest fingerprint as an input so the two cannot drift apart.
* ``retrieve``, ``generate`` -- not implemented yet.
"""

from __future__ import annotations

from quant_rag.config import PipelineConfig, load_config
from quant_rag.index.store import LoadedIndex, build_index, load_index
from quant_rag.ingest.run import IngestReport, ingest_corpus

__all__ = [
    "IngestReport",
    "LoadedIndex",
    "PipelineConfig",
    "build_index",
    "ingest_corpus",
    "load_config",
    "load_index",
]
