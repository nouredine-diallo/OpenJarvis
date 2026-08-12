"""Mission Engine asynchrone persistant (Phase 4 / D10).

Provides the durable, resumable mission primitive behind the JARVIS TEST:
a mission launched from a phone survives a server restart and can be resumed
from its last checkpoint.

Public surface:
* :class:`~openjarvis.missions.engine.MissionEngine` — background worker.
* :class:`~openjarvis.missions.store.MissionStore` — SQLite persistence.
* :func:`~openjarvis.missions.verifier.run_verification` — deterministic gate.
"""

from openjarvis.missions.engine import MissionEngine
from openjarvis.missions.store import MissionStore
from openjarvis.missions.types import (
    Mission,
    MissionEvent,
    MissionStatus,
    MissionStep,
    MissionStepStatus,
)
from openjarvis.missions.verifier import run_verification

__all__ = [
    "MissionEngine",
    "MissionStore",
    "Mission",
    "MissionEvent",
    "MissionStatus",
    "MissionStep",
    "MissionStepStatus",
    "run_verification",
]
