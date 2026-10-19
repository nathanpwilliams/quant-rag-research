import copy

import pytest
import yaml

from quant_rag.config import DEFAULT_CONFIG_PATH, ConfigError, PipelineConfig, load_config


@pytest.fixture
def raw() -> dict:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_committed_config_parses():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.experiment.name == "baseline_fixed_dense"
    assert config.ingest.parser == "pymupdf"
    assert config.retrieve.mode == "dense"
    assert config.index.bm25.enabled is False
    assert config.source_path == DEFAULT_CONFIG_PATH


def test_chunking_knobs_come_from_the_file(raw):
    """The ablation depends on these being read, never defaulted in code."""
    chunking = PipelineConfig.parse(raw).ingest.chunking
    # 510, not 512: the embedder's 512-token window counts [CLS] and [SEP], so
    # 512 content tokens would be silently truncated. See README.
    assert (chunking.strategy, chunking.chunk_tokens, chunking.overlap_tokens) == (
        "fixed",
        510,
        64,
    )

    modified = copy.deepcopy(raw)
    modified["ingest"]["chunking"].update(
        {"strategy": "structure", "chunk_tokens": 256, "overlap_tokens": 32}
    )
    changed = PipelineConfig.parse(modified).ingest.chunking
    assert (changed.strategy, changed.chunk_tokens, changed.overlap_tokens) == (
        "structure",
        256,
        32,
    )


def test_missing_knob_is_an_error_not_a_default(raw):
    modified = copy.deepcopy(raw)
    del modified["ingest"]["chunking"]["chunk_tokens"]
    with pytest.raises(ConfigError, match="chunk_tokens"):
        PipelineConfig.parse(modified)


def test_unknown_key_is_rejected(raw):
    """A typo must not silently leave the baseline value in force."""
    modified = copy.deepcopy(raw)
    modified["ingest"]["chunking"]["chunk_token"] = 256
    with pytest.raises(ConfigError, match="unknown key"):
        PipelineConfig.parse(modified)


def test_unknown_section_is_rejected(raw):
    modified = copy.deepcopy(raw)
    modified["rerank"] = {"enabled": True}
    with pytest.raises(ConfigError, match="unknown key"):
        PipelineConfig.parse(modified)


def test_unknown_strategy_is_rejected(raw):
    modified = copy.deepcopy(raw)
    modified["ingest"]["chunking"]["strategy"] = "semantic"
    with pytest.raises(ConfigError, match="strategy"):
        PipelineConfig.parse(modified)


@pytest.mark.parametrize("overlap", [512, 600])
def test_overlap_must_be_less_than_chunk(raw, overlap):
    """Equal would make the sliding window never advance."""
    modified = copy.deepcopy(raw)
    modified["ingest"]["chunking"]["overlap_tokens"] = overlap
    with pytest.raises(ConfigError, match="overlap_tokens"):
        PipelineConfig.parse(modified)


def test_bool_is_not_accepted_for_an_int_knob(raw):
    modified = copy.deepcopy(raw)
    modified["ingest"]["chunking"]["chunk_tokens"] = True
    with pytest.raises(ConfigError, match="got bool"):
        PipelineConfig.parse(modified)


def test_fingerprint_tracks_ingest_knobs_only(raw):
    base = PipelineConfig.parse(raw)
    assert base.ingest_fingerprint() == PipelineConfig.parse(copy.deepcopy(raw)).ingest_fingerprint()

    for section, key, value in [
        ("ingest", "parser", "pymupdf4llm"),
        ("retrieve", "mode", "hybrid"),
    ]:
        modified = copy.deepcopy(raw)
        modified[section][key] = value
        changed = PipelineConfig.parse(modified).ingest_fingerprint()
        if section == "ingest":
            assert changed != base.ingest_fingerprint()
        else:
            # Retrieval knobs do not change chunk content, so chunk sets are shared.
            assert changed == base.ingest_fingerprint()

    for key, value in [("strategy", "structure"), ("chunk_tokens", 256), ("overlap_tokens", 16)]:
        modified = copy.deepcopy(raw)
        modified["ingest"]["chunking"][key] = value
        assert PipelineConfig.parse(modified).ingest_fingerprint() != base.ingest_fingerprint()
