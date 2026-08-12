"""Markdown mirror of the 4-kind memory (Phase 3 — spec §6/§7).

Regenerates human-readable Markdown files (Obsidian-friendly) from the typed
fact store so the user can inspect and correct JARVIS's long-term memory:

- ``memory/preferences.md`` — Préférences
- ``memory/facts.md``      — Faits
- ``memory/decisions.md``  — Décisions
- ``memory/rules.md``      — Règles
- ``memory/PERSONAL_CONTEXT.md`` — combined compact block injected into every
  system prompt via ``SystemPromptBuilder`` so rules and preferences are
  actually applied without being repeated.

Every entry is dated and sourced; the mirror is read-only, regenerated from
the store on each mutation (never hand-edited).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from openjarvis.core.paths import get_config_dir
from openjarvis.memory.store import (
    Fact,
    KIND_DECISION,
    KIND_FACT,
    KIND_PREFERENCE,
    KIND_RULE,
    KINDS,
    normalize_kind,
)

logger = logging.getLogger(__name__)

KIND_LABELS = {
    KIND_PREFERENCE: "Préférences",
    KIND_FACT: "Faits",
    KIND_DECISION: "Décisions",
    KIND_RULE: "Règles",
}

KIND_FILES = {
    KIND_PREFERENCE: "preferences.md",
    KIND_FACT: "facts.md",
    KIND_DECISION: "decisions.md",
    KIND_RULE: "rules.md",
}

_COMBINED = "PERSONAL_CONTEXT.md"


def memory_mirror_dir() -> Path:
    """Directory holding the Markdown memory mirror."""
    return get_config_dir() / "memory"


def _entry_line(fact: Fact) -> str:
    ts = fact.created_at or 0.0
    if ts > 0:
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    else:
        date = "?"
    src = f" ({fact.source})" if fact.source else ""
    return f"- [{date}]{src} {fact.text}"


def _render_section(kind: str, facts: List[Fact]) -> str:
    label = KIND_LABELS.get(kind, kind)
    if not facts:
        return f"## {label}\n\n_(vide)_"
    lines = [f"## {label}", ""]
    # Newest first.
    for fact in sorted(facts, key=lambda f: f.created_at, reverse=True):
        lines.append(_entry_line(fact))
    return "\n".join(lines)


def refresh(
    store,
    base_dir: Path | None = None,
) -> List[Path]:
    """Regenerate the Markdown mirror from *store*. Returns written paths.

    Never raises: on failure the store data is untouched and we simply skip
    the mirror (it is a convenience view, not a source of truth).
    """
    paths: List[Path] = []
    try:
        facts = store.list()
    except Exception:  # noqa: BLE001
        logger.debug("Memory mirror refresh failed (store list)", exc_info=True)
        return paths

    base = Path(base_dir).expanduser() if base_dir is not None else memory_mirror_dir()
    try:
        base.mkdir(parents=True, exist_ok=True)
        grouped = {k: [f for f in facts if f.kind == normalize_kind(k)] for k in KINDS}
        for kind, fname in KIND_FILES.items():
            content = _render_section(kind, grouped[kind])
            path = base / fname
            path.write_text(content + "\n", encoding="utf-8")
            paths.append(path)

        # Combined compact block for system-prompt injection (rules first).
        combined = [
            "You have the following long-term memory about the user. "
            "Apply the Rules unconditionally, respect the Preferences, "
            "honor the Decisions, and ground answers in the Facts.",
            "",
        ]
        combined.append(_render_section(KIND_RULE, grouped[KIND_RULE]))
        combined.append("")
        combined.append(_render_section(KIND_PREFERENCE, grouped[KIND_PREFERENCE]))
        combined.append("")
        combined.append(_render_section(KIND_DECISION, grouped[KIND_DECISION]))
        combined.append("")
        combined.append(_render_section(KIND_FACT, grouped[KIND_FACT]))
        (base / _COMBINED).write_text("\n".join(combined) + "\n", encoding="utf-8")
        paths.append(base / _COMBINED)
    except Exception:  # noqa: BLE001
        logger.debug("Memory mirror refresh failed", exc_info=True)
    return paths


__all__ = [
    "KIND_LABELS",
    "KIND_FILES",
    "memory_mirror_dir",
    "refresh",
    "time",
]
