"""Tests for resolve_memory_backend -- the shared factory used by both
cli/serve.py and system/builder.py.

Exists because HybridMemory doesn't fit the generic
``MemoryRegistry.create(key, db_path=...)`` shape every other backend
does (it needs pre-built sparse=/dense= sub-backends): selecting
``default_backend = "hybrid"`` in config.toml silently raised a
TypeError at construction time before this factory existed -- found
live while implementing Brique 2 (docs/SPEC_BRIQUE2_MEMOIRE.md, which
recommends hybrid as the default).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastembed")
pytest.importorskip("sqlite_vec")

from openjarvis.core.config import StorageConfig  # noqa: E402
from openjarvis.core.registry import MemoryRegistry  # noqa: E402
from openjarvis.tools.storage import resolve_memory_backend  # noqa: E402
from openjarvis.tools.storage.hybrid import HybridMemory  # noqa: E402
from openjarvis.tools.storage.sqlite import SQLiteMemory  # noqa: E402
from openjarvis.tools.storage.sqlite_vec import SqliteVecMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _register_backends():
    """conftest clears every registry between tests -- re-register the
    backends this factory needs to resolve (same pattern as
    tests/memory/test_sqlite.py and test_hybrid.py)."""
    MemoryRegistry.register_value("sqlite", SQLiteMemory)
    MemoryRegistry.register_value("sqlite_vec", SqliteVecMemory)
    MemoryRegistry.register_value("hybrid", HybridMemory)


def _config(tmp_path: Path, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        memory=StorageConfig(default_backend=backend, db_path=str(tmp_path / "memory.db"))
    )


def test_unknown_backend_returns_none(tmp_path: Path):
    assert resolve_memory_backend(_config(tmp_path, "not-a-real-backend")) is None


def test_empty_backend_returns_none(tmp_path: Path):
    assert resolve_memory_backend(_config(tmp_path, "")) is None


def test_sqlite_backend_still_uses_generic_path(tmp_path: Path):
    """Non-hybrid backends are unaffected -- same construction as before."""
    backend = resolve_memory_backend(_config(tmp_path, "sqlite"))
    assert isinstance(backend, SQLiteMemory)
    backend.close()


def test_hybrid_backend_constructs_without_raising(tmp_path: Path):
    """This TypeError'd before resolve_memory_backend existed -- the whole
    point of this factory."""
    backend = resolve_memory_backend(_config(tmp_path, "hybrid"))
    assert isinstance(backend, HybridMemory)
    assert isinstance(backend._sparse, SQLiteMemory)
    assert isinstance(backend._dense, SqliteVecMemory)


def test_hybrid_sub_backends_use_distinct_db_files(tmp_path: Path):
    """The sparse and dense halves must not collide on the same sqlite
    file -- each needs its own schema."""
    backend = resolve_memory_backend(_config(tmp_path, "hybrid"))
    assert backend._sparse._db_path != backend._dense._db_path


def test_hybrid_backend_is_actually_usable(tmp_path: Path):
    """End-to-end: store through the factory-built hybrid backend, get it
    back via retrieve -- not just "doesn't crash on construction"."""
    backend = resolve_memory_backend(_config(tmp_path, "hybrid"))
    backend.store("Never deploy on a Friday", metadata={"kind": "rule"})
    results = backend.retrieve("Never deploy on a Friday", top_k=1)
    assert len(results) == 1
    assert "Friday" in results[0].content
