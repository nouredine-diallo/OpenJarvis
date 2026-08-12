"""REST API for the Mission Engine (Phase 4 / D10).

Endpoints mirror the mission lifecycle:

* ``POST /v1/missions``         → 202 + mission_id (async launch)
* ``GET  /v1/missions``         → list missions
* ``GET  /v1/missions/{id}``    → live state (resume point after a crash)
* ``GET  /v1/missions/{id}/events`` → immutable audit trail
* ``POST /v1/missions/{id}/pause|resume|cancel`` → lifecycle control

The engine is reached through ``request.app.state.mission_engine``; if it is
not present (missions disabled), the router returns 503.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from openjarvis.missions.types import MissionStep

router = APIRouter(prefix="/v1/missions", tags=["missions"])


def _engine(request: Request):
    engine = getattr(request.app.state, "mission_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Mission engine is disabled")
    return engine


class _StepIn(BaseModel):
    title: str
    prompt: str = ""
    required_capabilities: List[str] = []


class _MissionIn(BaseModel):
    goal: str
    kind: str = ""
    autonomy_level: Optional[int] = None
    max_steps: Optional[int] = None
    max_budget_tokens: Optional[int] = None
    steps: Optional[List[_StepIn]] = None
    requested_by: str = ""


@router.post("")
async def create_mission(body: _MissionIn, request: Request):
    """Launch a mission asynchronously. Returns 202 + mission_id.

    ``kind="coding_pr"`` auto-plans a coding mission as 5 checkpointed steps
    (setup/implement/test/review/ship) instead of one mega-step; ignored if
    ``steps`` is given explicitly.
    """
    engine = _engine(request)
    steps = None
    if body.steps:
        steps = [
            MissionStep(
                index=i,
                title=s.title,
                prompt=s.prompt,
                required_capabilities=s.required_capabilities,
            )
            for i, s in enumerate(body.steps)
        ]
    mission = engine.launch(
        body.goal,
        steps=steps,
        kind=body.kind,
        autonomy_level=body.autonomy_level,
        max_steps=body.max_steps,
        max_budget_tokens=body.max_budget_tokens,
        requested_by=body.requested_by,
    )
    return {
        "status": "accepted",
        "mission_id": mission.mission_id,
        "mission_status": mission.status,
    }


@router.get("")
async def list_missions(
    request: Request, status: Optional[str] = None, limit: int = 100
):
    """List missions, optionally filtered by status."""
    engine = _engine(request)
    missions = engine._store.list_missions(status=status, limit=limit)
    return {
        "missions": [
            {"mission_id": m.mission_id, "goal": m.goal, "status": m.status}
            for m in missions
        ]
    }


@router.get("/{mission_id}")
async def get_mission(mission_id: str, request: Request):
    """Return the live mission state (the exact resume point)."""
    engine = _engine(request)
    mission = engine.status(mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return mission.to_dict()


@router.get("/{mission_id}/events")
async def get_mission_events(
    request: Request, mission_id: str, limit: int = 500
):
    """Return the immutable audit trail of a mission."""
    engine = _engine(request)
    if engine.status(mission_id) is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    events = engine._store.list_events(mission_id, limit=limit)
    return {"mission_id": mission_id, "events": [e.to_dict() for e in events]}


@router.post("/{mission_id}/pause")
async def pause_mission(mission_id: str, request: Request):
    engine = _engine(request)
    mission = engine.pause(mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"mission_id": mission_id, "status": mission.status}


@router.post("/{mission_id}/resume")
async def resume_mission(mission_id: str, request: Request):
    engine = _engine(request)
    mission = engine.resume(mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"mission_id": mission_id, "status": mission.status}


@router.post("/{mission_id}/cancel")
async def cancel_mission(mission_id: str, request: Request):
    engine = _engine(request)
    mission = engine.cancel(mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"mission_id": mission_id, "status": mission.status}
