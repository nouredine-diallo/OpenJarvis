"""Storage primitive — persistent searchable storage."""

from __future__ import annotations

from typing import Any, Optional

from openjarvis.core.registry import MemoryRegistry

# Always-available backend
import openjarvis.tools.storage.sqlite  # noqa: F401

# Optional backends — import to trigger registration
try:
    import openjarvis.tools.storage.bm25  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.tools.storage.faiss_backend  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.tools.storage.colbert_backend  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.tools.storage.hybrid  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.tools.storage.dense  # noqa: F401
except ImportError:
    pass

try:
    import openjarvis.tools.storage.sqlite_vec  # noqa: F401
except ImportError:
    pass

from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult
from openjarvis.tools.storage.chunking import Chunk, ChunkConfig, chunk_text
from openjarvis.tools.storage.context import ContextConfig, inject_context
from openjarvis.tools.storage.ingest import ingest_path, read_document


def resolve_memory_backend(config: Any) -> Optional[MemoryBackend]:
    """Build the configured ``config.memory.default_backend`` retrieval
    backend, or ``None`` if disabled/unresolvable.

    Every backend other than ``"hybrid"`` takes a single ``db_path`` kwarg,
    so ``MemoryRegistry.create(key, db_path=...)`` was enough on its own --
    that's what both ``cli/serve.py`` and ``system/builder.py`` did inline
    before this function existed. ``HybridMemory`` doesn't fit that shape
    (it needs pre-built ``sparse=``/``dense=`` sub-backend instances), so
    selecting ``default_backend = "hybrid"`` in config.toml silently raised
    a ``TypeError`` at construction time -- found while implementing Brique
    2 (docs/SPEC_BRIQUE2_MEMOIRE.md), which recommends hybrid as the
    default: dense-only retrieval measurably misses short rule-like entries
    (spec §3.3), so hybrid isn't an optional nicety here.

    This is the one place that knows how to build every registered
    backend, hybrid included; both call sites should use it instead of
    reimplementing the special case.
    """
    key = getattr(config.memory, "default_backend", "")
    if not key or not MemoryRegistry.contains(key):
        return None

    if key != "hybrid":
        return MemoryRegistry.create(key, db_path=config.memory.db_path)

    from pathlib import Path

    from openjarvis.tools.storage.hybrid import HybridMemory
    from openjarvis.tools.storage.sqlite import SQLiteMemory
    from openjarvis.tools.storage.sqlite_vec import SqliteVecMemory

    base = Path(config.memory.db_path)
    sparse = SQLiteMemory(db_path=str(base))
    dense = SqliteVecMemory(db_path=str(base.with_name(base.stem + "_vec" + base.suffix)))
    # dense_weight=2.0, not the default 1.0/1.0: measured live on a 20-query
    # FR paraphrase benchmark (docs/SPEC_BRIQUE2_MEMOIRE.md follow-up) where
    # queries deliberately share no keywords with the stored text -- equal
    # weighting let BM25's noise on those queries drag fused accuracy below
    # dense-alone (35% vs 45% top-1); 2.0 recovers most of that (40%) while
    # keeping sparse able to win the cases it's actually good at (exact
    # terms, short rule-like entries -- the spec's original "hybrid
    # obligatoire" finding). Neither figure is great in absolute terms --
    # short-text semantic retrieval is a genuinely hard problem, not fully
    # solved by this first pass; a reranker is a plausible future
    # improvement, deliberately out of scope here.
    return HybridMemory(sparse=sparse, dense=dense, dense_weight=2.0)


__all__ = [
    "Chunk",
    "ChunkConfig",
    "ContextConfig",
    "MemoryBackend",
    "RetrievalResult",
    "chunk_text",
    "inject_context",
    "ingest_path",
    "read_document",
    "resolve_memory_backend",
]
