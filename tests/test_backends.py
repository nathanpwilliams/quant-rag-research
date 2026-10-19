"""Search-backend tests.

Two things need proving. First, that ``exact_numpy`` really is exact — it is
the default, so every eval number comes through it. Second, that the two
backends agree, which is what makes the choice an implementation detail rather
than an experimental variable.

The faiss tests are skipped if faiss-cpu is absent, and are the only tests that
load it. That is the point of the default: the ordinary path never imports it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_rag.index.backends import ExactNumpyBackend, build_backend


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors)


@pytest.fixture
def vectors() -> np.ndarray:
    # Corpus-shaped: 461 chunks of 768 dims, as the real index.
    return unit_vectors(461, 768)


@pytest.fixture
def queries() -> np.ndarray:
    return unit_vectors(8, 768, seed=1)


# --------------------------------------------------------------------------- #
# exact_numpy is exact
# --------------------------------------------------------------------------- #


def test_exact_numpy_matches_a_brute_force_reference(vectors, queries):
    backend = ExactNumpyBackend(vectors)
    scores, rows = backend.search(queries, 10)

    reference = queries @ vectors.T
    expected_rows = np.argsort(-reference, axis=1, kind="stable")[:, :10]
    assert (rows == expected_rows).all()
    assert scores == pytest.approx(np.take_along_axis(reference, rows, axis=1))


def test_exact_numpy_scores_descend_per_query(vectors, queries):
    scores, _rows = ExactNumpyBackend(vectors).search(queries, 10)
    for row in scores:
        assert list(row) == sorted(row, reverse=True)


def test_exact_numpy_self_match_is_rank_one_at_score_one(vectors):
    """A normalized vector's nearest neighbour is itself, at cosine 1."""
    backend = ExactNumpyBackend(vectors)
    scores, rows = backend.search(vectors[:20], 1)
    assert (rows[:, 0] == np.arange(20)).all()
    assert scores[:, 0] == pytest.approx(1.0, abs=1e-5)


def test_exact_numpy_clamps_k_to_the_corpus_and_never_pads(vectors, queries):
    backend = ExactNumpyBackend(unit_vectors(4, 768))
    scores, rows = backend.search(queries, 50)
    assert rows.shape == (8, 4)
    assert scores.shape == (8, 4)
    assert (rows >= 0).all()


def test_exact_numpy_breaks_ties_toward_the_lower_row(queries):
    """Duplicate vectors score identically; the order must not be arbitrary."""
    base = unit_vectors(1, 8)
    # Rows 0, 1, 2 are identical, so all three tie for any query.
    vectors = np.ascontiguousarray(np.vstack([base, base, base, unit_vectors(1, 8, 9)]))
    backend = ExactNumpyBackend(vectors)
    _scores, rows = backend.search(base, 3)
    assert list(rows[0]) == [0, 1, 2]
    # Deterministic across repeated calls, not merely sorted once.
    for _ in range(5):
        assert list(backend.search(base, 3)[1][0]) == [0, 1, 2]


def test_exact_numpy_rejects_mismatched_query_dim(vectors):
    backend = ExactNumpyBackend(vectors)
    with pytest.raises(ValueError, match="does not match index dim"):
        backend.search(unit_vectors(2, 16), 5)


def test_exact_numpy_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D vector array"):
        ExactNumpyBackend(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError, match="2-D queries"):
        ExactNumpyBackend(unit_vectors(4, 8)).search(np.zeros(8, dtype=np.float32), 2)


def test_exact_numpy_reports_shape(vectors):
    backend = ExactNumpyBackend(vectors)
    assert backend.n_vectors == 461
    assert backend.dim == 768
    assert backend.name == "exact_numpy"


def test_build_backend_dispatches_and_rejects_unknown_names(vectors):
    assert isinstance(build_backend(vectors, "exact_numpy"), ExactNumpyBackend)
    with pytest.raises(ValueError, match="unknown index.backend"):
        build_backend(vectors, "annoy")


# --------------------------------------------------------------------------- #
# The two backends agree, so the choice is not an experimental variable
# --------------------------------------------------------------------------- #


@pytest.fixture
def faiss_backend(vectors):
    pytest.importorskip(
        "quant_rag.index._runtime", reason="faiss-cpu not installed"
    )  # installs the OpenMP guard; never import faiss bare
    from quant_rag.index.backends import FaissFlatIPBackend

    return FaissFlatIPBackend(vectors)


def test_the_two_backends_return_the_same_top_k(vectors, queries, faiss_backend):
    numpy_scores, numpy_rows = ExactNumpyBackend(vectors).search(queries, 10)
    faiss_scores, faiss_rows = faiss_backend.search(queries, 10)

    assert (numpy_rows == faiss_rows).all(), (
        "backends disagree on ranking; they are supposed to be the same "
        "exhaustive search"
    )
    assert numpy_scores == pytest.approx(faiss_scores, abs=1e-5)


def test_faiss_backend_reports_the_same_shape(vectors, faiss_backend):
    assert faiss_backend.n_vectors == 461
    assert faiss_backend.dim == 768
    assert faiss_backend.name == "faiss_flat_ip"


def test_faiss_backend_clamps_k_so_neither_backend_pads(vectors, queries):
    pytest.importorskip(
        "quant_rag.index._runtime", reason="faiss-cpu not installed"
    )  # installs the OpenMP guard; never import faiss bare
    from quant_rag.index.backends import FaissFlatIPBackend

    small = unit_vectors(4, 768)
    _scores, rows = FaissFlatIPBackend(small).search(queries, 50)
    assert rows.shape == (8, 4)
    assert (rows >= 0).all()


def test_faiss_backend_rejects_unsupported_index_types(vectors):
    pytest.importorskip(
        "quant_rag.index._runtime", reason="faiss-cpu not installed"
    )  # installs the OpenMP guard; never import faiss bare
    from quant_rag.index.backends import FaissFlatIPBackend

    with pytest.raises(ValueError, match="index_type"):
        FaissFlatIPBackend(vectors, index_type="hnsw")
