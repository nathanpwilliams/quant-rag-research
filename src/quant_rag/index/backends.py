"""Search backends: exact numpy (default) and FAISS flat IP (optional).

Both are *exact* searches over the same stored vectors — they score every chunk
and return the true top-k, so they are interchangeable and produce the same
ranking. The backend is a search strategy, not a storage format: vectors are
persisted once as ``vectors.npy`` and whichever backend is configured is
constructed from them at load time. Switching backends therefore never requires
re-embedding, which is why ``index.backend`` is not part of the index
fingerprint.

Why ``exact_numpy`` is the default, and not a workaround
--------------------------------------------------------

This corpus is 461 chunks of 768 float32 — a 1.4 MB matrix. A query is one
``(1, 768) @ (768, 461)`` matmul: ~350k multiply-adds, microseconds under BLAS,
with no index structure to build, tune, or persist. FAISS's value is
approximate search over millions of vectors; at this scale ``IndexFlatIP`` does
exactly the same brute-force computation numpy does, just behind a native
extension. So numpy is not a lesser substitute here — it is the same algorithm
with fewer moving parts.

It also removes a real failure, not a hypothetical one. ``faiss-cpu`` and
``torch`` each bundle their own ``libomp.dylib`` with different install names,
so macOS dyld loads two OpenMP runtimes and the process dies once both are
used: ``OMP: Error #15`` (SIGABRT), or SIGSEGV inside
``__kmp_suspend_initialize_thread`` on an OpenMP worker thread once
``KMP_DUPLICATE_LIB_OK`` is set to silence the abort. ``quant-rag ask`` has to
embed a query (torch) and search the index in one process, so the default path
must not import faiss at all. ``exact_numpy`` doesn't.

``faiss_flat_ip`` stays available for comparison and for a future corpus that
actually needs ANN, and routes through :mod:`quant_rag.index._runtime`, which
installs the OpenMP guard. See that module for what the guard does and does not
make safe.

Tie-breaking
------------

Both backends sort descending by score and break exact ties toward the lower
row index, which is the lower chunk ordinal. Eval numbers should not move
because two chunks scored identically and a sort was unstable, so
``exact_numpy`` sorts with an explicitly stable kind rather than relying on the
default.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SearchBackend(Protocol):
    """Exact top-k search over a fixed matrix of chunk vectors."""

    name: str

    @property
    def n_vectors(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``(scores, rows)``, each ``(n_queries, min(k, n_vectors))``.

        Rows index into the vector matrix in chunk-set order. Scores are raw
        inner products, descending per query; no sentinel padding, so a caller
        never has to filter ``-1``.
        """


class ExactNumpyBackend:
    """Brute-force inner product via one matmul. The default backend."""

    name = "exact_numpy"

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D vector array, got shape {vectors.shape}")
        # float32 matches what the embedder produces and what FAISS would use,
        # so the two backends agree to the bit rather than to a tolerance.
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def n_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError(f"expected 2-D queries, got shape {queries.shape}")
        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim {queries.shape[1]} does not match index dim {self.dim}"
            )
        width = min(k, self.n_vectors)
        scores = queries @ self.vectors.T
        # Stable sort on the negated scores: descending by score, and among
        # equal scores the lower row index first. argpartition would be faster
        # asymptotically but is unstable, and at this size the full sort of 461
        # values is already negligible.
        rows = np.argsort(-scores, axis=1, kind="stable")[:, :width]
        return np.take_along_axis(scores, rows, axis=1), rows


class FaissFlatIPBackend:
    """``faiss.IndexFlatIP``: the same exhaustive search, via FAISS.

    Optional. Importing faiss alongside torch is what the OpenMP guard in
    :mod:`quant_rag.index._runtime` exists for, so this backend pulls it in
    through that module and nowhere else.
    """

    name = "faiss_flat_ip"

    def __init__(self, vectors: np.ndarray, index_type: str = "flat_ip"):
        from quant_rag.index._runtime import faiss  # noqa: PLC0415 - see module docstring

        if index_type != "flat_ip":
            raise ValueError(f"unsupported index.faiss.index_type: {index_type!r}")
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D vector array, got shape {vectors.shape}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index: Any = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._dim = int(vectors.shape[1])

    @property
    def n_vectors(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return self._dim

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim {queries.shape[1]} does not match index dim {self.dim}"
            )
        width = min(k, self.n_vectors)
        scores, rows = self.index.search(queries, width)
        # Asking for exactly ntotal or fewer means FAISS never pads with -1,
        # so the two backends return the same shape for the same k.
        return scores, rows


BACKENDS = {
    "exact_numpy": ExactNumpyBackend,
    "faiss_flat_ip": FaissFlatIPBackend,
}


def build_backend(vectors: np.ndarray, backend: str, *, index_type: str = "flat_ip") -> SearchBackend:
    """Construct the backend named by ``index.backend``."""
    if backend == "exact_numpy":
        return ExactNumpyBackend(vectors)
    if backend == "faiss_flat_ip":
        return FaissFlatIPBackend(vectors, index_type=index_type)
    raise ValueError(f"unknown index.backend: {backend!r}")  # pragma: no cover
