"""Agent tool for the 4-kind long-term memory (Phase 3).

Lets the agent explicitly store and query preferences, facts, decisions and
rules — complementing the automatic background extraction. Writes refresh the
Markdown mirror immediately.
"""

from __future__ import annotations

import logging
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.memory.mirror import refresh as _refresh_mirror
from openjarvis.memory.store import KIND_FACT, KINDS, create_fact_store, normalize_kind
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

_KIND_HINT = ", ".join(f'"{k}"' for k in KINDS)


@ToolRegistry.register("personal_memory")
class PersonalMemoryTool(BaseTool):
    """Store or query the user's long-term memory (4 kinds)."""

    tool_id = "personal_memory"
    is_local = True

    def __init__(self, store: Any | None = None):
        self._store = store if store is not None else self._default_store()

    @staticmethod
    def _default_store():
        try:
            from openjarvis.core.config import load_config

            mem = load_config().memory
            return create_fact_store(
                getattr(mem, "backend", "local"),
                path=getattr(mem, "facts_path", None),
                max_facts=getattr(mem, "max_facts", 1000),
            )
        except Exception:  # noqa: BLE001
            return create_fact_store()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="personal_memory",
            description=(
                "Store or query JARVIS's long-term memory about the user. "
                "Kinds: "
                "preference (stable user preference), fact (durable dated fact), "
                "decision (locked decision), rule (convention JARVIS must follow). "
                "Store rules/preferences to remember them permanently and apply "
                "them in every future conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "list"],
                        "description": "store: save an entry. list: show entries.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(KINDS),
                        "description": f"Memory kind ({_KIND_HINT}).",
                    },
                    "text": {
                        "type": "string",
                        "description": "Entry text (required for action=store).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return (list only).",
                    },
                },
                "required": ["action", "kind"],
            },
            category="memory",
        )

    def execute(self, **params: Any) -> ToolResult:
        action = str(params.get("action", "")).strip().lower()
        kind = normalize_kind(params.get("kind"))
        if action == "store":
            text = str(params.get("text", "")).strip()
            if not text:
                return ToolResult(
                    tool_name="personal_memory",
                    content="No text provided for action=store.",
                    success=False,
                )
            stored = self._store.add(text, source="agent", kind=kind)
            _refresh_mirror(self._store)
            if stored:
                return ToolResult(
                    tool_name="personal_memory",
                    content=f"Stored {kind}: {text}",
                    success=True,
                )
            return ToolResult(
                tool_name="personal_memory",
                content=f"Already stored (duplicate): {text}",
                success=False,
            )

        if action == "list":
            limit = max(1, int(params.get("limit", 50)))
            entries = self._store.list(kind=kind)[-limit:]
            if not entries:
                return ToolResult(
                    tool_name="personal_memory",
                    content=f"No '{kind}' entries in memory.",
                    success=True,
                )
            lines = [f"{kind} memory ({len(entries)}):"]
            for i, fact in enumerate(entries, 1):
                src = f" [{fact.source}]" if fact.source else ""
                lines.append(f"{i}. {fact.text}{src}")
            return ToolResult(
                tool_name="personal_memory",
                content="\n".join(lines),
                success=True,
            )

        return ToolResult(
            tool_name="personal_memory",
            content=f"Unknown action '{action}'. Use 'store' or 'list'.",
            success=False,
        )


__all__ = ["PersonalMemoryTool"]
