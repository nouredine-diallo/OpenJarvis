"""Tests for the background memory service (openjarvis.memory.service)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from openjarvis.core.config import StorageConfig
from openjarvis.core.events import EventBus
from openjarvis.memory.service import (
    MemoryService,
    build_memory_service,
    publish_completed_exchange,
)
from openjarvis.memory.store import LocalFactStore


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeExtractor:
    """Extractor stub with controllable output, blocking and failures."""

    def __init__(self, facts=None, *, raises=None, gate=None):
        self._facts = facts or []
        self._raises = raises
        self._gate = gate  # optional threading.Event to block on
        self.calls = []

    def extract(self, user_text, assistant_text=""):
        self.calls.append((user_text, assistant_text))
        if self._gate is not None:
            self._gate.wait(timeout=2.0)
        if self._raises is not None:
            raise self._raises
        return list(self._facts)


def _service(tmp_path, extractor, **kwargs):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    return MemoryService(store, extractor, **kwargs)


def test_start_stop_lifecycle(tmp_path):
    svc = _service(tmp_path, FakeExtractor())
    assert svc.is_running is False
    svc.start()
    assert svc.is_running is True
    svc.start()  # idempotent
    assert svc.is_running is True
    svc.stop()
    assert svc.is_running is False
    svc.stop()  # idempotent


def test_submit_extracts_and_stores(tmp_path):
    extractor = FakeExtractor(["User likes hiking"])
    svc = _service(tmp_path, extractor)
    svc.start()
    try:
        assert svc.submit("I love hiking", "Nice!") is True
        assert _wait_until(lambda: svc.fact_count() == 1)
        assert [f.text for f in svc.list_facts()] == ["User likes hiking"]
    finally:
        svc.stop()


def test_completed_exchange_event_extracts_and_stores(tmp_path):
    bus = EventBus(record_history=True)
    extractor = FakeExtractor(["User likes jazz"])
    store = LocalFactStore(tmp_path / "facts.jsonl")
    svc = MemoryService(store, extractor, event_bus=bus)
    svc.start()
    try:
        assert publish_completed_exchange(
            bus,
            "I like jazz",
            "Noted.",
            source="test",
        )
        assert _wait_until(lambda: svc.fact_count() == 1)
        assert extractor.calls == [("I like jazz", "Noted.")]
    finally:
        svc.stop()


def test_completed_exchange_event_unsubscribes_on_stop(tmp_path):
    bus = EventBus(record_history=True)
    extractor = FakeExtractor(["User likes jazz"])
    store = LocalFactStore(tmp_path / "facts.jsonl")
    svc = MemoryService(store, extractor, event_bus=bus)
    svc.start()
    svc.stop()

    publish_completed_exchange(bus, "I like jazz", "Noted.", source="test")

    assert extractor.calls == []


def test_submit_when_not_running_is_dropped(tmp_path):
    extractor = FakeExtractor(["x"])
    svc = _service(tmp_path, extractor)
    assert svc.submit("hi", "there") is False
    assert extractor.calls == []


def test_submit_empty_user_text_dropped(tmp_path):
    extractor = FakeExtractor(["x"])
    svc = _service(tmp_path, extractor)
    svc.start()
    try:
        assert svc.submit("   ", "y") is False
    finally:
        svc.stop()


def test_worker_survives_extractor_broken_pipe(tmp_path):
    """A BrokenPipeError in one job must not kill the worker."""
    extractor = FakeExtractor(raises=BrokenPipeError("client gone"))
    svc = _service(tmp_path, extractor)
    svc.start()
    try:
        svc.submit("first", "a")
        assert _wait_until(lambda: len(extractor.calls) == 1)
        # Service is still alive and accepting work.
        assert svc.is_running is True
        assert svc.submit("second", "b") is True
        assert _wait_until(lambda: len(extractor.calls) == 2)
    finally:
        svc.stop()


def test_worker_survives_generic_exception(tmp_path):
    extractor = FakeExtractor(raises=RuntimeError("boom"))
    svc = _service(tmp_path, extractor)
    svc.start()
    try:
        svc.submit("x", "y")
        assert _wait_until(lambda: len(extractor.calls) == 1)
        assert svc.is_running is True
    finally:
        svc.stop()


def test_submit_returns_false_when_queue_full(tmp_path):
    """Backpressure: a full queue drops work instead of blocking the caller."""
    gate = threading.Event()
    extractor = FakeExtractor(["fact"], gate=gate)
    svc = _service(tmp_path, extractor, max_queue=1)
    svc.start()
    try:
        # First submit is pulled by the worker and blocks on the gate.
        assert svc.submit("job1", "a") is True
        assert _wait_until(lambda: len(extractor.calls) == 1)
        # Fill the (size-1) queue, then the next submit must be dropped.
        assert svc.submit("job2", "b") is True
        dropped = svc.submit("job3", "c")
        assert dropped is False
    finally:
        gate.set()
        svc.stop()


def test_build_memory_service_disabled_returns_none(tmp_path):
    cfg = SimpleNamespace(memory=StorageConfig(enabled=False))
    assert build_memory_service(cfg, object(), "model") is None


def test_build_memory_service_no_engine_returns_none(tmp_path):
    cfg = SimpleNamespace(memory=StorageConfig(enabled=True))
    assert build_memory_service(cfg, None, "model") is None


def test_build_memory_service_no_model_returns_none(tmp_path):
    cfg = SimpleNamespace(memory=StorageConfig(enabled=True, extraction_model=""))
    assert build_memory_service(cfg, object(), "") is None


def test_build_memory_service_enabled(tmp_path):
    cfg = SimpleNamespace(
        memory=StorageConfig(
            enabled=True,
            extraction_model="qwen3:14b",
            facts_path=str(tmp_path / "facts.jsonl"),
            max_facts=10,
        )
    )
    svc = build_memory_service(cfg, object(), "fallback-model")
    assert isinstance(svc, MemoryService)


def test_build_memory_service_falls_back_to_default_model(tmp_path):
    cfg = SimpleNamespace(
        memory=StorageConfig(
            enabled=True,
            extraction_model="",
            facts_path=str(tmp_path / "facts.jsonl"),
        )
    )
    svc = build_memory_service(cfg, object(), "active-model")
    assert isinstance(svc, MemoryService)


class FakeRetrievalBackend:
    """Records every store() call -- test double for the fact store <->
    retrieval backend link (Brique 2, spec §4.2 point 4)."""

    def __init__(self, *, raises: bool = False):
        self.stored = []
        self._raises = raises

    def store(self, content, *, source="", metadata=None):
        if self._raises:
            raise RuntimeError("backend down")
        self.stored.append((content, source, metadata or {}))
        return "fake-id"


class TypedFactExtractor:
    """Stub exposing extract_typed (the Fact-returning path), matching
    the real FactExtractor's typed API rather than the plain-string one."""

    def __init__(self, facts):
        from openjarvis.memory.store import Fact

        self._facts = [f if isinstance(f, Fact) else Fact(text=f) for f in facts]

    def extract_typed(self, user_text, assistant_text=""):
        return list(self._facts)


def test_facts_are_indexed_into_retrieval_backend(tmp_path):
    from openjarvis.memory.store import Fact, KIND_RULE

    retrieval = FakeRetrievalBackend()
    extractor = TypedFactExtractor([Fact(text="Never deploy on Friday", kind=KIND_RULE)])
    svc = _service(tmp_path, extractor, retrieval_backend=retrieval)
    svc.start()
    try:
        svc.submit("don't deploy Fridays", "noted")
        assert _wait_until(lambda: len(retrieval.stored) == 1)
        content, source, meta = retrieval.stored[0]
        assert content == "Never deploy on Friday"
        assert meta["kind"] == KIND_RULE
        assert "created_at" in meta
    finally:
        svc.stop()


def test_no_retrieval_backend_configured_is_a_silent_noop(tmp_path):
    """Default behavior (no retrieval_backend passed) must be unchanged --
    this is what every existing MemoryService caller does today."""
    extractor = FakeExtractor(["User likes hiking"])
    svc = _service(tmp_path, extractor)  # no retrieval_backend kwarg
    svc.start()
    try:
        svc.submit("I love hiking", "Nice!")
        assert _wait_until(lambda: svc.fact_count() == 1)
    finally:
        svc.stop()


def test_retrieval_backend_failure_does_not_break_fact_store(tmp_path):
    """A broken retrieval backend must never take down fact-store writes --
    the fact store stays the source of truth regardless."""
    from openjarvis.memory.store import Fact

    retrieval = FakeRetrievalBackend(raises=True)
    extractor = TypedFactExtractor([Fact(text="Some durable fact")])
    svc = _service(tmp_path, extractor, retrieval_backend=retrieval)
    svc.start()
    try:
        svc.submit("tell me something", "ok")
        assert _wait_until(lambda: svc.fact_count() == 1)
        assert [f.text for f in svc.list_facts()] == ["Some durable fact"]
    finally:
        svc.stop()


def test_build_memory_service_passes_retrieval_backend_through(tmp_path):
    retrieval = FakeRetrievalBackend()
    cfg = SimpleNamespace(
        memory=StorageConfig(
            enabled=True,
            extraction_model="active-model",
            facts_path=str(tmp_path / "facts.jsonl"),
        )
    )
    svc = build_memory_service(cfg, object(), retrieval_backend=retrieval)
    assert svc._retrieval_backend is retrieval
