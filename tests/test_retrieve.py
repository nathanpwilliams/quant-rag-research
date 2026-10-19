"""Dense retrieval tests.

The sanity test that matters is the last one: a query quoting a chunk verbatim
must rank that chunk first. It uses the real BGE model against the real index,
because a stand-in embedder can only prove the plumbing, not that the
query/passage asymmetry and the scoring are actually right. It skips when the
corpus or the index is absent, since both are gitignored.
"""

from __future__ import annotations

import pytest

from quant_rag.index.store import IndexMissingError, IndexPairingError, load_index
from quant_rag.retrieve.ask import format_hit, preview
from quant_rag.retrieve.dense import DenseRetriever, Hit, build_retriever
from tests.test_index import FakeEmbedder, built, make_config, write_chunks  # noqa: F401


# --------------------------------------------------------------------------- #
# Plumbing, on the deterministic stand-in embedder
# --------------------------------------------------------------------------- #


def test_search_returns_ranked_hits_with_descending_scores(built):
    _config, chunks, index, _directory, embedder = built
    hits = DenseRetriever(index, embedder).search("triple-barrier labeling", 4)

    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(isinstance(h, Hit) for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {h.chunk_id for h in hits} == {c.chunk_id for c in chunks}


def test_search_respects_k(built):
    _config, _chunks, index, _directory, embedder = built
    assert len(DenseRetriever(index, embedder).search("q", 2)) == 2


def test_search_caps_at_corpus_size_and_drops_the_faiss_sentinel(built):
    """Asking for more than exists must not invent hits from row -1."""
    _config, chunks, index, _directory, embedder = built
    hits = DenseRetriever(index, embedder).search("q", 50)
    assert len(hits) == len(chunks)
    assert all(h.chunk is not None for h in hits)


def test_search_rejects_nonpositive_k(built):
    _config, _chunks, index, _directory, embedder = built
    with pytest.raises(ValueError, match="k must be positive"):
        DenseRetriever(index, embedder).search("q", 0)


def test_query_instruction_is_applied_to_queries_only(built):
    """The BGE asymmetry: prefix on the query, never on the passage."""
    _config, chunks, index, _directory, embedder = built
    DenseRetriever(index, embedder).search("what is the EIIE topology", 1)

    assert embedder.encoded_queries[-1] == "Q: what is the EIIE topology"
    assert all(not text.startswith("Q: ") for text in embedder.encoded_passages)
    assert embedder.encoded_passages == [c.text for c in chunks]


def test_empty_query_instruction_leaves_the_query_untouched(tmp_path):
    config = make_config(tmp_path, query_instruction="")
    write_chunks(config, [("d", "alpha beta"), ("d", "gamma delta")])
    embedder = FakeEmbedder(config.index.embedding)
    from quant_rag.index.store import build_index

    index, _directory = build_index(config, embedder=embedder, log=lambda _m: None)
    DenseRetriever(index, embedder).search("alpha beta", 1)
    assert embedder.encoded_queries[-1] == "alpha beta"


def test_retriever_refuses_an_embedder_that_did_not_build_the_index(built):
    """A query embedded by another model scores silently wrong; make it loud."""
    config, _chunks, index, _directory, _embedder = built
    other = FakeEmbedder(make_config(_directory, model="different-model").index.embedding)
    with pytest.raises(ValueError, match="does not match the index"):
        DenseRetriever(index, other)


def test_retriever_refuses_a_mismatched_normalize_setting(built):
    config, _chunks, index, _directory, _embedder = built
    other = FakeEmbedder(make_config(_directory, normalize=False).index.embedding)
    with pytest.raises(ValueError, match="normalize"):
        DenseRetriever(index, other)


def test_metric_is_reported_from_the_index_metadata(built):
    _config, _chunks, index, _directory, embedder = built
    assert DenseRetriever(index, embedder).metric == "cosine"


def test_build_retriever_refuses_unimplemented_modes(tmp_path):
    for mode in ("sparse", "hybrid"):
        config = make_config(tmp_path, mode=mode)
        with pytest.raises(NotImplementedError, match=mode):
            build_retriever(config)


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #


def test_preview_collapses_whitespace_and_truncates():
    assert preview("a\n\nb   c") == "a b c"
    long = "x " * 400
    out = preview(long, 200)
    assert len(out) == 203 and out.endswith("...")


def test_format_hit_shows_the_fields_the_cli_promises(built):
    _config, _chunks, index, _directory, embedder = built
    hit = DenseRetriever(index, embedder).search("triple-barrier labeling", 1)[0]
    rendered = format_hit(hit)
    assert hit.doc_id in rendered
    assert hit.chunk_id in rendered
    assert f"{hit.score:+.4f}" in rendered
    assert preview(hit.chunk.body) in rendered


# --------------------------------------------------------------------------- #
# Retrieval sanity against the real model and the real corpus index
# --------------------------------------------------------------------------- #

REAL_CONFIG = "config/pipeline.yaml"


@pytest.fixture
def real_retriever():
    from quant_rag.config import load_config

    config = load_config(REAL_CONFIG)
    try:
        index = load_index(config)
    except IndexMissingError:
        pytest.skip("no index on disk; run `uv run quant-rag index` first")
    except IndexPairingError as exc:
        pytest.fail(f"index on disk does not match its chunk set: {exc}")

    pytest.importorskip("sentence_transformers")
    from quant_rag.index.embed import Embedder

    try:
        embedder = Embedder(config.index.embedding)
    except Exception as exc:  # noqa: BLE001 - no model cache, no network
        pytest.skip(f"embedding model unavailable: {exc}")
    return DenseRetriever(index, embedder)


def eiie_chunk(retriever):
    """The chunk of the portfolio paper that carries the EIIE definition."""
    return next(
        c
        for c in retriever.index.chunks
        if c.doc_id == "arxiv_1706.10059"
        and "Ensemble of Identical Independent Evaluators" in c.body
    )


def test_a_verbatim_phrase_retrieves_its_own_chunk_first(real_retriever):
    """The end-to-end sanity check: quote a chunk, get that chunk back at rank 1.

    The phrase is read out of the index rather than hard-coded, so the test
    keeps its meaning when chunk boundaries move.

    It is taken from the chunk's *opening*, which is deliberate and not an
    accident of convenience: BGE pools with the CLS token
    (``modules.json`` -> ``pooling_mode: cls``), so a long passage's vector is
    dominated by how it starts. A phrase from the opening is therefore what
    this property actually guarantees; see
    ``test_mid_chunk_evidence_is_diluted_by_a_512_token_chunk`` for the
    measured limit on phrases from further in.
    """
    target = eiie_chunk(real_retriever)
    phrase = " ".join(target.body.split()[:40])
    assert len(phrase.split()) == 40, "need a distinctive phrase to make this meaningful"

    hits = real_retriever.search(phrase, 5)
    assert hits[0].chunk_id == target.chunk_id, (
        f"expected {target.chunk_id} at rank 1, got "
        f"{[(h.chunk_id, round(h.score, 4)) for h in hits]}"
    )
    assert hits[0].score > 0.8, f"verbatim opening scored only {hits[0].score:.4f}"


def test_full_chunk_text_is_its_own_nearest_neighbour(real_retriever):
    """Identity check across the index: a chunk queried by its own full text
    must come back first, at near-1 cosine.

    This is the strongest statement that the plumbing lines up end to end --
    the embedder that built the vectors, the stored matrix, and the row ->
    chunk_id mapping. An off-by-one anywhere in that chain breaks it.
    """
    chunks = real_retriever.index.chunks
    # Spread across the corpus: first, middle, last.
    for chunk in (chunks[0], chunks[len(chunks) // 2], chunks[-1]):
        hits = real_retriever.search(chunk.text, 1)
        assert hits[0].chunk_id == chunk.chunk_id, (
            f"{chunk.chunk_id} queried by its own text returned "
            f"{hits[0].chunk_id}"
        )
        assert hits[0].score > 0.9, (
            f"{chunk.chunk_id} self-similarity only {hits[0].score:.4f}"
        )


def test_mid_chunk_evidence_is_diluted_by_a_512_token_chunk(real_retriever):
    """A measured property of the baseline row, not a passing grade.

    A 51-word phrase quoted verbatim from the *middle* of a 512-token chunk
    does not retrieve that chunk first -- CLS pooling means the chunk's vector
    is dominated by its opening, so a differently-topiced chunk can outrank the
    one that literally contains the text. Recall@10 still holds, and that is
    what this asserts; closing the rank-1 gap is precisely what the
    structure-aware chunking row is meant to test.

    Asserted as a contract (top 10) rather than a frozen rank, so an
    improvement does not fail the suite but a regression below recall@10 does.
    """
    target = eiie_chunk(real_retriever)
    middle = len(target.body) // 2
    phrase = " ".join(target.body[middle : middle + 400].split()[1:-1])

    hits = real_retriever.search(phrase, 10)
    ranks = {h.chunk_id: h.rank for h in hits}
    assert target.chunk_id in ranks, (
        f"{target.chunk_id} contains the phrase verbatim but is not in the top "
        f"10: {[(h.chunk_id, round(h.score, 4)) for h in hits]}"
    )


def test_scores_are_cosine_similarities_in_range(real_retriever):
    hits = real_retriever.search("How do these papers prevent lookahead bias?", 10)
    assert hits
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert real_retriever.metric == "cosine"


def test_a_topical_query_retrieves_the_document_that_owns_the_topic(real_retriever):
    """Triple-barrier labeling exists in exactly one corpus document."""
    hits = real_retriever.search("triple-barrier labeling", 5)
    assert hits[0].doc_id == "note_afml_labeling", [
        (h.doc_id, round(h.score, 4)) for h in hits
    ]
