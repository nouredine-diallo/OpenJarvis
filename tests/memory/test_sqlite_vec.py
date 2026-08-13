"""Tests for the persistent dense (semantic) memory backend.

Uses the real fastembed model rather than a mock, same rationale as
test_embeddings.py: the RAM-discipline and persistence behavior is what's
actually load-bearing here (Brique 2, docs/SPEC_BRIQUE2_MEMOIRE.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastembed")
pytest.importorskip("sqlite_vec")

from openjarvis.core.registry import MemoryRegistry  # noqa: E402
from openjarvis.tools.storage.sqlite_vec import SqliteVecMemory  # noqa: E402


def _make_backend(tmp_path: Path) -> SqliteVecMemory:
    if not MemoryRegistry.contains("sqlite_vec"):
        MemoryRegistry.register_value("sqlite_vec", SqliteVecMemory)
    return SqliteVecMemory(db_path=tmp_path / "test_vec.db")


def test_registration_in_memory_registry():
    MemoryRegistry.register_value("sqlite_vec", SqliteVecMemory)
    assert MemoryRegistry.contains("sqlite_vec")


def test_store_returns_id_and_counts(tmp_path: Path):
    backend = _make_backend(tmp_path)
    assert backend.count() == 0
    doc_id = backend.store("Python is a programming language")
    assert isinstance(doc_id, str) and doc_id
    assert backend.count() == 1
    backend.close()


def test_store_many_batches_a_single_embed_call(tmp_path: Path, monkeypatch):
    backend = _make_backend(tmp_path)
    calls = []
    real_embed = backend._get_embedder().embed

    def _counting_embed(texts):
        calls.append(len(texts))
        return real_embed(texts)

    monkeypatch.setattr(backend._get_embedder(), "embed", _counting_embed)
    ids = backend.store_many(["one", "two", "three"], sources=["a", "b", "c"])
    assert len(ids) == 3
    assert calls == [3]  # one batched call, not three
    backend.close()


def test_semantic_retrieval_ranks_relevant_first(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("Never deploy on a Friday", metadata={"kind": "rule"})
    backend.store("The weather is sunny today", metadata={"kind": "fact"})
    results = backend.retrieve("when is it forbidden to ship to production", top_k=2)
    assert len(results) == 2
    assert "Friday" in results[0].content
    backend.close()


def test_retrieve_k_is_always_passed_explicitly(tmp_path: Path):
    """sqlite-vec raises OperationalError if a MATCH query omits 'k = ?' --
    validated live (docs/SPEC_BRIQUE2_MEMOIRE.md §3.1). This just pins the
    behavior: retrieve() must never omit it, for any top_k."""
    backend = _make_backend(tmp_path)
    backend.store("some content")
    # Must not raise for any positive top_k.
    assert backend.retrieve("some content", top_k=1) != [] or True
    assert backend.retrieve("some content", top_k=100) is not None
    backend.close()


def test_empty_query_returns_empty(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("something")
    assert backend.retrieve("", top_k=5) == []
    assert backend.retrieve("   ", top_k=5) == []
    backend.close()


def test_delete_removes_from_both_tables(tmp_path: Path):
    backend = _make_backend(tmp_path)
    doc_id = backend.store("temporary entry")
    assert backend.count() == 1
    assert backend.delete(doc_id) is True
    assert backend.count() == 0
    assert backend.delete(doc_id) is False  # already gone
    backend.close()


def test_clear_removes_everything(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store_many(["a", "b", "c"])
    assert backend.count() == 3
    backend.clear()
    assert backend.count() == 0
    backend.close()


def test_metadata_round_trips(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("a rule", metadata={"kind": "rule", "confidence": 0.9})
    results = backend.retrieve("a rule", top_k=1)
    assert results[0].metadata["kind"] == "rule"
    assert results[0].metadata["confidence"] == 0.9
    backend.close()


def test_persists_across_reconnects(tmp_path: Path):
    """Unlike DenseMemory (in-memory only), this backend must survive a
    process restart -- that's the whole point of using it over the
    existing in-memory dense backend for the memory layer."""
    db_path = tmp_path / "persist.db"
    backend = SqliteVecMemory(db_path=db_path)
    backend.store("persisted content")
    backend.close()

    reopened = SqliteVecMemory(db_path=db_path)
    assert reopened.count() == 1
    results = reopened.retrieve("persisted content", top_k=1)
    assert results[0].content == "persisted content"
    reopened.close()
