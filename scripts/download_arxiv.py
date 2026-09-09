#!/usr/bin/env python3
"""Thin wrapper around ``quant_rag.ingest.download`` (also: ``quant-rag download``).

Runs with a bare interpreter, no third-party deps:
    python3 scripts/download_arxiv.py --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_rag.ingest.download import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
