"""Stage wiring: ingest -> index -> retrieve -> generate.

Each stage reads its knobs from the ``config/*.yaml`` file passed on the CLI;
an ablation row is a config file, never a code branch. Stage implementations
land in their own packages (``ingest/``, ``index/``, ``retrieve/``,
``generate/``); this module only composes them.

Stage status:

* ``ingest`` -- implemented: :func:`quant_rag.ingest.run.ingest_corpus` parses
  the manifest's documents to ``data/processed/parsed/`` and writes a chunk set
  keyed by :meth:`quant_rag.config.PipelineConfig.ingest_fingerprint`.
* ``index``, ``retrieve``, ``generate`` -- not implemented yet.
"""

from __future__ import annotations

from quant_rag.config import PipelineConfig, load_config
from quant_rag.ingest.run import IngestReport, ingest_corpus

__all__ = ["IngestReport", "PipelineConfig", "ingest_corpus", "load_config"]
