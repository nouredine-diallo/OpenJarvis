"""Persistent stores for automatically extracted long-term memory facts.

A *fact* is a short, durable statement worth remembering about the user
(e.g. ``"Prefers concise answers"``).  Facts are produced by the memory
service's background extractor and persisted here so they survive across
sessions.  The store is intentionally small and self-contained: it dedupes,
caps the total number of facts, and is safe to call from multiple threads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import FactStoreRegistry


def _default_fact_path() -> Path:
    """Return the env-aware default JSONL path for automatic memory facts."""
    return get_config_dir() / "memory_facts.jsonl"


# Memory kinds (spec §6/§8: Préférence ≠ Fait ≠ Décision ≠ Règle).
KIND_FACT = "fact"
KIND_PREFERENCE = "preference"
KIND_DECISION = "decision"
KIND_RULE = "rule"
KINDS: tuple = (KIND_FACT, KIND_PREFERENCE, KIND_DECISION, KIND_RULE)


def normalize_kind(kind: str | None) -> str:
    """Return a canonical kind, defaulting unknown/empty to ``"fact"``."""
    k = (kind or KIND_FACT).strip().lower()
    if k in KINDS:
        return k
    return KIND_FACT


@dataclass(slots=True)
class Fact:
    """A single durable memory entry."""

    text: str
    source: str = ""
    created_at: float = 0.0
    kind: str = KIND_FACT


class FactStore(ABC):
    """Abstract persistent store for extracted memory facts."""

    @abstractmethod
    def add(self, text: str, source: str = "", kind: str = KIND_FACT) -> bool:
        """Store *text* as a memory of the given *kind*.

        Returns True if a new entry was stored.
        """

    def add_many(
        self,
        texts: Iterable[str],
        source: str = "",
        kind: str = KIND_FACT,
    ) -> int:
        """Store several entries, returning the count of newly stored ones."""
        added = 0
        for text in texts:
            if self.add(text, source=source, kind=kind):
                added += 1
        return added

    def add_facts(
        self,
        facts: "Iterable[Fact]",
        source: str = "",
    ) -> int:
        """Store typed entries (kind + text), returning newly stored count.

        *source* is used as a fallback for entries that carry no explicit
        source of their own.
        """
        added = 0
        for fact in facts:
            if self.add(fact.text, source=fact.source or source, kind=fact.kind):
                added += 1
        return added

    @abstractmethod
    def list(self, kind: str | None = None) -> List[Fact]:
        """Return all stored entries, optionally filtered by *kind*."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all stored facts, returning the number removed."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored facts."""


@FactStoreRegistry.register("local")
class LocalFactStore(FactStore):
    """Append-only JSONL fact store on the local filesystem.

    Facts are kept human-readable (one JSON object per line) so they can be
    inspected or edited by hand.  Writes are atomic (temp file + rename) and
    guarded by a lock, so concurrent ``add`` calls from the extraction worker
    and ``list``/``clear`` from the CLI never corrupt the file.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_facts: int = 1000,
    ) -> None:
        self._path = (
            Path(path).expanduser() if path is not None else _default_fact_path()
        )
        self._max_facts = max(0, int(max_facts))
        self._lock = threading.Lock()
        self._facts: List[Fact] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> List[Fact]:
        if not self._path.exists():
            return []
        facts: List[Fact] = []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # skip malformed lines rather than crashing
            fact_text = str(obj.get("text", "")).strip()
            if not fact_text:
                continue
            facts.append(
                Fact(
                    text=fact_text,
                    source=str(obj.get("source", "")),
                    created_at=float(obj.get("created_at", 0.0) or 0.0),
                    kind=normalize_kind(obj.get("kind")),
                )
            )
        return facts

    def _flush(self) -> None:
        """Atomically rewrite the JSONL file from the in-memory list."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = "".join(
            json.dumps(asdict(f), ensure_ascii=False) + "\n" for f in self._facts
        )
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self._path)

    def _sync_from_disk_locked(self) -> None:
        """Refresh in-memory facts from disk while holding ``self._lock``."""
        self._facts = self._load()

    # -- FactStore API ------------------------------------------------------

    def add(self, text: str, source: str = "", kind: str = KIND_FACT) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        kind = normalize_kind(kind)
        with self._lock:
            self._sync_from_disk_locked()
            lowered = text.lower()
            if any(f.text.lower() == lowered for f in self._facts):
                return False  # dedupe
            self._facts.append(
                Fact(text=text, source=source, created_at=time.time(), kind=kind)
            )
            # Enforce the cap by evicting the oldest entries.
            if self._max_facts and len(self._facts) > self._max_facts:
                self._facts = self._facts[-self._max_facts :]
            self._flush()
        return True

    def list(self, kind: str | None = None) -> List[Fact]:
        with self._lock:
            self._sync_from_disk_locked()
            if kind is None:
                return list(self._facts)
            k = normalize_kind(kind)
            return [f for f in self._facts if f.kind == k]

    def clear(self) -> int:
        with self._lock:
            self._sync_from_disk_locked()
            removed = len(self._facts)
            self._facts = []
            if self._path.exists():
                try:
                    self._path.unlink()
                except OSError:
                    self._flush()
        return removed

    def count(self) -> int:
        with self._lock:
            self._sync_from_disk_locked()
            return len(self._facts)

    @property
    def path(self) -> Path:
        """Filesystem location of the JSONL store."""
        return self._path


def _ensure_fact_store_backends_registered() -> None:
    """Restore built-in fact-store registrations if a test cleared registries."""
    if not FactStoreRegistry.contains("local"):
        FactStoreRegistry.register_value("local", LocalFactStore)


def create_fact_store(
    backend: str = "local",
    *,
    path: str | Path | None = None,
    max_facts: int = 1000,
) -> FactStore:
    """Construct a fact store for the configured *backend*.

    Only the ``"local"`` (on-disk JSONL) backend is supported today; the
    registry-backed constructor exists so additional backends can be added
    without changing the service or CLI wiring.
    """
    _ensure_fact_store_backends_registered()
    key = (backend or "local").strip().lower()
    if not FactStoreRegistry.contains(key):
        supported = ", ".join(FactStoreRegistry.keys())
        raise ValueError(
            f"Unknown memory backend '{backend}'. Supported backends: {supported}"
        )
    return FactStoreRegistry.create(key, path, max_facts=max_facts)


__all__ = [
    "Fact",
    "FactStore",
    "LocalFactStore",
    "create_fact_store",
    "KIND_FACT",
    "KIND_PREFERENCE",
    "KIND_DECISION",
    "KIND_RULE",
    "KINDS",
    "normalize_kind",
]
