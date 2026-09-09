"""Stage wiring: ingest -> index -> retrieve -> generate.

Each stage reads its knobs from the ``config/*.yaml`` file passed on the CLI;
an ablation row is a config file, never a code branch. Stage implementations
land in their own packages (``ingest/``, ``index/``, ``retrieve/``,
``generate/``); this module only composes them.
"""
