"""Agent tools for the Mission Engine (Phase 4 / D10).

The agent (e.g. via Telegram) can launch a durable, resumable mission and
ask about its progress.  The engine instance is injected post-build through
the class attribute ``_mission_engine`` (same pattern as the scheduler tools),
so instance attribute lookup finds it at call time even for instances created
before the engine existed.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("launch_mission")
class LaunchMissionTool(BaseTool):
    """Launch a durable, resumable mission that runs asynchronously."""

    tool_id = "launch_mission"
    _mission_engine: Optional[Any] = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="launch_mission",
            description=(
                "Launch a new mission (a durable, resumable background task). "
                "The mission keeps running even if the chat is closed, survives "
                "a server restart, and reports back when done. Use this for "
                "long-running tasks like 'recherche sur le web et fais un "
                "rapport'. Returns a mission_id to track progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The objective of the mission.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["default", "coding_pr"],
                        "description": (
                            "'coding_pr' auto-plans a coding mission that "
                            "ends in an open PR as 5 checkpointed steps "
                            "(setup/implement/test/review/ship) instead of "
                            "one big step -- use this for any request that "
                            "implies writing code, fixing a bug, or opening "
                            "a PR. 'default' (or omitted) for everything "
                            "else (research, single-shot tasks)."
                        ),
                    },
                    "autonomy_level": {
                        "type": "integer",
                        "description": "autonomy 0-5; 0=approve each step, "
                        ">=1 autonomous (default 1).",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum steps (BudgetGuard cap).",
                    },
                },
                "required": ["goal"],
            },
        )

    def execute(self, **params: Any) -> ToolResult:
        engine = self._mission_engine
        if engine is None:
            return ToolResult(
                tool_name=self.tool_id,
                content="Mission engine is not available on this server.",
                success=False,
            )
        goal = str(params.get("goal", "")).strip()
        if not goal:
            return ToolResult(
                tool_name=self.tool_id,
                content="A mission needs a goal.",
                success=False,
            )
        try:
            mission = engine.launch(
                goal,
                kind=str(params.get("kind", "") or ""),
                autonomy_level=params.get("autonomy_level"),
                max_steps=params.get("max_steps"),
                requested_by=params.get("_channel", "") or params.get("requested_by", ""),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Failed to launch mission: {exc}",
                success=False,
            )
        return ToolResult(
            tool_name=self.tool_id,
            content=(
                f"Mission lancée ! ID : {mission.mission_id} "
                f"(autonomie {mission.autonomy_level}, max {mission.max_steps} étapes). "
                f"Je te préviens quand elle est terminée. "
                f"Demande 'où en est ma mission {mission.mission_id}?' pour le suivi."
            ),
            metadata={"mission_id": mission.mission_id},
        )


@ToolRegistry.register("mission_status")
class MissionStatusTool(BaseTool):
    """Report the live status and audit trail of a mission."""

    tool_id = "mission_status"
    _mission_engine: Optional[Any] = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="mission_status",
            description=(
                "Check the status of a mission by its mission_id. Returns the "
                "current step, progress, and latest result so the user can "
                "'find their mission again' after closing the chat."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mission_id": {
                        "type": "string",
                        "description": "The mission_id returned at launch.",
                    }
                },
                "required": ["mission_id"],
            },
        )

    def execute(self, **params: Any) -> ToolResult:
        engine = self._mission_engine
        mission_id = str(params.get("mission_id", "")).strip()
        if engine is None:
            return ToolResult(
                tool_name=self.tool_id,
                content="Mission engine is not available on this server.",
                success=False,
            )
        if not mission_id:
            return ToolResult(
                tool_name=self.tool_id,
                content="You must provide a mission_id.",
                success=False,
            )
        mission = engine.status(mission_id)
        if mission is None:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Mission {mission_id} introuvable.",
                success=False,
            )
        return ToolResult(
            tool_name=self.tool_id,
            content=_format_status(mission),
            metadata={"mission_id": mission_id, "status": mission.status},
        )


def _format_status(mission) -> str:
    lines = [
        f"Mission {mission.mission_id} — {mission.status}",
        f"Objectif : {mission.goal}",
    ]
    for step in mission.steps:
        icon = {"succeeded": "OK", "running": "…", "pending": "·", "failed": "X"}.get(
            step.status, "·"
        )
        lines.append(f"  [{icon}] Étape {step.index}: {step.title}")
        if step.result and step.status == "succeeded":
            lines.append(f"      {step.result[:200]}")
        if step.error:
            lines.append(f"      erreur: {step.error[:200]}")
    if mission.report:
        lines.append("")
        lines.append(mission.report[:800])
    return "\n".join(lines)


__all__ = ["LaunchMissionTool", "MissionStatusTool"]
