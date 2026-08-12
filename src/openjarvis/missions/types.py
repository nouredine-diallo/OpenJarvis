"""Mission domain types (Phase 4 — Mission Engine asynchrone persistant).

A *mission* is a durable, resumable unit of asynchronous work.  It is the
concrete carrier of the "JARVIS TEST" scenario: launched from a phone while the
server may be off, it must be retrievable identically after a restart.

The mission lifecycle is::

    PENDING -> RUNNING -> SUCCEEDED
                 |  \\--> PAUSED -> RUNNING
                 |  \\--> WAITING_FOR_WORKER -> RUNNING   (D12)
                 \\--> FAILED
                 \\--> CANCELLED

A mission ``WAITING_FOR_WORKER`` has been accepted but some step needs a
capability the current worker lacks (Docker, browser, GPU…); it resumes
automatically when a capable worker registers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Autonomy ladder (spec §6/§8, PLAN.md Phase 4).
AUTONOMY_LEVELS = (0, 1, 2, 3, 4, 5)


class MissionStatus(str, Enum):
    """Lifecycle states of a mission."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_WORKER = "waiting_for_worker"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def inflight(cls) -> frozenset["MissionStatus"]:
        """Statuses that mean the mission still has work to do."""
        return frozenset(
            {cls.PENDING, cls.RUNNING, cls.PAUSED, cls.WAITING_FOR_WORKER}
        )


class MissionStepStatus(str, Enum):
    """Lifecycle states of a single mission step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class MissionStep:
    """A single atomic unit of work inside a mission."""

    index: int
    title: str
    prompt: str = ""
    status: str = MissionStepStatus.PENDING.value
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    retries: int = 0
    required_capabilities: List[str] = field(default_factory=list)
    # Soft preference for the "heavy reasoning" agent tier (e.g. the user's
    # Claude subscription CLI) when one is configured -- unlike
    # required_capabilities this never blocks the mission (WAITING_FOR_WORKER)
    # if unavailable; the engine just falls back to the normal coding agent
    # or system. Used for steps where quality matters most (self-review).
    prefer_heavy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class Mission:
    """A durable, resumable unit of asynchronous work."""

    mission_id: str
    goal: str
    created_at: float
    updated_at: float
    status: str = MissionStatus.PENDING.value
    autonomy_level: int = 1
    max_steps: int = 10
    max_budget_tokens: int = 50000
    budget_used_tokens: int = 0
    current_step: int = 0
    steps: List[MissionStep] = field(default_factory=list)
    report: str = ""
    artefacts: List[str] = field(default_factory=list)
    requested_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)

    # -- conversions --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mission":
        steps = [MissionStep.from_dict(s) for s in data.get("steps", [])]
        d = {k: v for k, v in data.items() if k != "steps"}
        d["steps"] = steps
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # -- helpers -------------------------------------------------------------

    def next_step(self) -> Optional[MissionStep]:
        """Return the next step that has not succeeded, or ``None``."""
        for step in self.steps:
            if step.status != MissionStepStatus.SUCCEEDED.value:
                return step
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            MissionStatus.SUCCEEDED.value,
            MissionStatus.FAILED.value,
            MissionStatus.CANCELLED.value,
        )


@dataclass(slots=True)
class MissionEvent:
    """An immutable audit record describing one mission mutation."""

    mission_id: str
    ts: float
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


__all__ = [
    "AUTONOMY_LEVELS",
    "MissionStatus",
    "MissionStepStatus",
    "Mission",
    "MissionStep",
    "MissionEvent",
]
