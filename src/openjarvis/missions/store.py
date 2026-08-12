"""SQLite persistence for missions and their immutable audit log.

Pure-Python ``sqlite3`` (no Rust extension required), mirroring the
``SchedulerStore`` pattern.  The full mission state (including every step and
its result) is stored as a JSON document on each mutation — this is the
*checkpoint* that makes crash recovery possible: restarting the server and
re-reading the row restores the mission at the exact step it stopped at.

Every state change also appends a row to ``mission_events``, the audit trail
that lets JARVIS answer "où en est ma mission ?" with proof.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.missions.types import Mission, MissionEvent, MissionStatus

_CREATE_MISSIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS missions (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    goal        TEXT NOT NULL,
    updated_at  REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
"""

_CREATE_EVENTS_TABLE = """\
CREATE TABLE IF NOT EXISTS mission_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id  TEXT NOT NULL,
    ts          REAL NOT NULL,
    event_type  TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mission_events_mission ON mission_events(mission_id);
"""


class MissionStore:
    """SQLite CRUD store for missions and their audit events."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_MISSIONS_TABLE)
        self._conn.executescript(_CREATE_EVENTS_TABLE)
        self._conn.commit()

    # -- missions -----------------------------------------------------------

    def create_mission(self, mission: Mission) -> None:
        """Persist a brand-new mission (idempotent by mission_id)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO missions (id, status, goal, updated_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    mission.mission_id,
                    mission.status,
                    mission.goal,
                    mission.updated_at,
                    json.dumps(mission.to_dict()),
                ),
            )
            self._conn.commit()

    def save_mission(self, mission: Mission) -> None:
        """Checkpoint the full mission state (atomic write of the JSON doc)."""
        mission.updated_at = max(mission.updated_at, _now())
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO missions (id, status, goal, updated_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    mission.mission_id,
                    mission.status,
                    mission.goal,
                    mission.updated_at,
                    json.dumps(mission.to_dict()),
                ),
            )
            self._conn.commit()

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        """Return one mission, or ``None`` if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM missions WHERE id = ?", (mission_id,)
            ).fetchone()
        if row is None:
            return None
        return Mission.from_dict(json.loads(row["data"]))

    def list_missions(
        self,
        status: Optional[str] = None,
        *,
        limit: int = 100,
    ) -> List[Mission]:
        """Return missions, optionally filtered by status, newest first."""
        with self._lock:
            if status is not None:
                rows = self._conn.execute(
                    "SELECT data FROM missions WHERE status = ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data FROM missions ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [Mission.from_dict(json.loads(r["data"])) for r in rows]

    def list_inflight(self) -> List[Mission]:
        """Return missions that still have work to do (pending/running/paused)."""
        placeholders = ", ".join("?" for _ in MissionStatus.inflight())
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM missions WHERE status IN (%s) "
                "ORDER BY updated_at ASC" % placeholders,
                tuple(MissionStatus.inflight()),
            ).fetchall()
        return [Mission.from_dict(json.loads(r["data"])) for r in rows]

    def delete_mission(self, mission_id: str) -> None:
        """Remove a mission and its events (used by tests/cleanup)."""
        with self._lock:
            self._conn.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
            self._conn.execute(
                "DELETE FROM mission_events WHERE mission_id = ?", (mission_id,)
            )
            self._conn.commit()

    # -- audit events ---------------------------------------------------------

    def append_event(self, event: MissionEvent) -> None:
        """Append an immutable audit record."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO mission_events (mission_id, ts, event_type, data) "
                "VALUES (?, ?, ?, ?)",
                (
                    event.mission_id,
                    event.ts,
                    event.event_type,
                    json.dumps(event.data),
                ),
            )
            self._conn.commit()

    def list_events(self, mission_id: str, *, limit: int = 500) -> List[MissionEvent]:
        """Return the audit trail for a mission, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT mission_id, ts, event_type, data FROM mission_events "
                "WHERE mission_id = ? ORDER BY id ASC LIMIT ?",
                (mission_id, limit),
            ).fetchall()
        return [
            MissionEvent(
                mission_id=r["mission_id"],
                ts=r["ts"],
                event_type=r["event_type"],
                data=json.loads(r["data"]),
            )
            for r in rows
        ]

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def path(self) -> str:
        return self._db_path


def _now() -> float:
    import time

    return time.time()


__all__ = ["MissionStore"]
