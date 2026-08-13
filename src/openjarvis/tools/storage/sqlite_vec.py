"""Persistent dense (semantic) memory backend using ``sqlite-vec``.

Replaces :class:`~openjarvis.tools.storage.dense.DenseMemory` for the
semantic memory layer (Brique 2, docs/SPEC_BRIQUE2_MEMOIRE.md):
``DenseMemory`` is in-memory only (rebuilt at startup, fine for the small
docs corpus it was built for) and defaults to :class:`OllamaEmbedder`
(Ollama abandoned this project, see PLAN.md D9). This backend persists to
disk like :class:`~openjarvis.tools.storage.sqlite.SQLiteMemory` does for
the sparse side, and defaults to
:class:`~openjarvis.tools.storage.embeddings.FastEmbedEmbedder` (no
server, no torch).

API validated live against a real ``sqlite-vec`` install (/tmp/b2,
docs/SPEC_BRIQUE2_MEMOIRE.md §3.1): ``k`` must always be passed explicitly
in the ``MATCH`` query -- omitting it raises
``OperationalError: A LIMIT or 'k = ?' constraint is required``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult
from openjarvis.tools.storage.embeddings import Embedder, FastEmbedEmbedder


@MemoryRegistry.register("sqlite_vec")
class SqliteVecMemory(MemoryBackend):
    """Persistent semantic retrieval backend via SQLite + the ``vec0``
    virtual table extension.

    Parameters
    ----------
    db_path:
        Where to persist the SQLite database. Defaults alongside the
        sparse backend's ``memory.db`` (as ``memory_vec.db``) so the two
        halves of a hybrid setup live next to each other.
    embedder:
        An :class:`Embedder`. Lazily constructed as
        :class:`FastEmbedEmbedder` on first use if omitted, so
        instantiating this class never triggers a model download.
    """

    backend_id = "sqlite_vec"

    def __init__(
        self,
        db_path: str | Path = "",
        *,
        embedder: Optional[Embedder] = None,
    ) -> None:
        if not db_path:
            from openjarvis.core.config import DEFAULT_CONFIG_DIR

            db_path = str(DEFAULT_CONFIG_DIR / "memory_vec.db")
        self._db_path = str(db_path)
        self._embedder = embedder
        self._lock = threading.Lock()
        self._conn = None  # type: ignore[assignment]
        self._dim: Optional[int] = None

    # -- lazy setup -----------------------------------------------------

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedEmbedder()
        return self._embedder

    def _get_conn(self):
        if self._conn is not None:
            return self._conn
        import sqlite3

        import sqlite_vec

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        dim = self._get_embedder().dim()
        self._dim = dim
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index "
            f"USING vec0(embedding float[{dim}])"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vec_docs (
                rowid      INTEGER PRIMARY KEY,
                doc_id     TEXT UNIQUE NOT NULL,
                content    TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT '',
                metadata   TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        self._conn = conn
        return conn

    # -- MemoryBackend ABC -----------------------------------------------

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.store_many([content], sources=[source], metadatas=[metadata or {}])[0]

    def store_many(
        self,
        contents: List[str],
        *,
        sources: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not contents:
            return []
        sources = sources if sources is not None else [""] * len(contents)
        metadatas = metadatas if metadatas is not None else [{} for _ in contents]

        vectors = self._get_embedder().embed(contents)
        doc_ids = [uuid.uuid4().hex for _ in contents]
        now = time.time()

        with self._lock:
            conn = self._get_conn()
            for doc_id, content, source, meta, vector in zip(
                doc_ids, contents, sources, metadatas, vectors
            ):
                cur = conn.execute(
                    "INSERT INTO vec_docs (doc_id, content, source, metadata, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, content, source, json.dumps(meta), now),
                )
                conn.execute(
                    "INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, vector.tobytes()),
                )
            conn.commit()
        return doc_ids

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        if not query or not query.strip() or top_k <= 0:
            return []

        q_vec = self._get_embedder().embed([query])[0]

        with self._lock:
            conn = self._get_conn()
            # k must always be passed explicitly -- omitting it raises
            # OperationalError (validated live, see module docstring).
            rows = conn.execute(
                """
                SELECT d.content, d.source, d.metadata, d.doc_id, v.distance
                FROM vec_index v
                JOIN vec_docs d ON d.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (q_vec.tobytes(), top_k),
            ).fetchall()

        results: List[RetrievalResult] = []
        for content, source, meta_json, doc_id, distance in rows:
            meta = json.loads(meta_json) if meta_json else {}
            meta["doc_id"] = doc_id
            # L2 distance on normalized vectors -> similarity in [-1, 1],
            # consistent with the cosine scores DenseMemory/RRF expect.
            similarity = 1.0 - (float(distance) ** 2) / 2.0
            results.append(
                RetrievalResult(content=content, score=similarity, source=source, metadata=meta)
            )
        return results

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT rowid FROM vec_docs WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return False
            rowid = row[0]
            conn.execute("DELETE FROM vec_docs WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))
            conn.commit()
        return True

    def clear(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM vec_docs")
            conn.execute("DELETE FROM vec_index")
            conn.commit()

    def count(self) -> int:
        with self._lock:
            conn = self._get_conn()
            return int(conn.execute("SELECT COUNT(*) FROM vec_docs").fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = ["SqliteVecMemory"]
