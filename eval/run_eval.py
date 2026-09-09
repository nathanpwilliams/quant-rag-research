"""Golden-set evaluation runner (owner-authored metric logic lives here).

Planned flow:
1. Load ``eval/golden_set.jsonl`` and the active pipeline config.
2. Resolve each ``gold_evidence`` locator to chunk IDs under the ACTIVE chunking
   config (locators are verbatim phrases, so they survive re-chunking).
3. Run retrieval for every question; compute recall@k and MRR per tier.
4. Optionally run generation + RAGAS metrics.
5. Append a results CSV under ``eval/results/`` (append-only, never overwrite).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.parse_args(argv)
    raise SystemExit("eval/run_eval.py: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
