"""The single place torch and FAISS are imported, because they conflict.

``faiss-cpu`` and ``torch`` each bundle their own ``libomp.dylib`` with a
different install name (``/DLC/faiss/.dylibs/libomp.dylib`` vs
``/opt/llvm-openmp/lib/libomp.dylib``), so dyld loads them as two separate
OpenMP runtimes and the second one to initialize aborts the process:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

Import order does not help -- the abort is on the second runtime's init, so it
fires whichever way round the imports go once both libraries are actually used.
``quant-rag ask`` must embed a query (torch) and search an index (faiss) in one
process, so this has to be handled, not sequenced around.

What is done here, and why each piece is load-bearing:

1. ``KMP_DUPLICATE_LIB_OK=TRUE`` downgrades the abort to a warning. Alone this
   is *not* safe: LLVM documents it as unsupported and able to "silently
   produce incorrect results", and it really does bite -- with only this set,
   a multithreaded ``torch`` matmul in a process that had already used faiss
   segfaulted during development. Hence 2.
2. **Both** runtimes are pinned to one thread. The hazard is two OpenMP
   runtimes each building a thread pool and contending; with neither creating
   a pool, there is no contention to corrupt anything. Pinning only faiss is
   what left the segfault above, so ``torch`` is pinned too.
3. The pins and faiss's numerical output are asserted in
   ``tests/test_index.py`` (``test_both_openmp_runtimes_are_pinned``,
   ``test_faiss_search_matches_exact_numpy_search``), in a process with both
   runtimes loaded. "Silently incorrect" only stays silent if nobody looks.

Cost of the pin: embedding is one-time and batched, and on this project's
default ``device: auto`` it runs on MPS/CUDA where the heavy matmuls are not
OpenMP work anyway, so the measured difference is negligible. On a CPU-only
machine the index build is slower; that is traded for not shipping a crash.

A faster alternative exists but is not reproducible through ``uv sync``:
pointing ``faiss/.dylibs/libomp.dylib`` at torch's copy leaves exactly one
runtime and permits full threading. It is deliberately not done here, because
it is venv surgery that any re-sync silently undoes -- which would turn a
reproducible repo into one that works only on a hand-patched machine.

Everything that needs either library imports it from here.
"""

from __future__ import annotations

import os

# Must precede both imports: libomp reads this when it initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
import faiss  # noqa: E402

# Point 2. Set once, at import, so no call site can forget.
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

__all__ = ["faiss", "torch"]
