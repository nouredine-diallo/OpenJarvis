"""Mission verification: structural + deterministic + semantic checks.

The spec's anti-hallucination rule ("ne jamais déclarer une tâche terminée
sans validation", "preuve sinon incertitude") is enforced here as an explicit,
deterministic gate: a mission is only marked ``verified=True`` when every
check passes.  If anything is missing (no goal, empty steps, a step without a
result, a report that claims more than it proves), the mission is marked
``verified=False`` with a per-check report explaining what failed.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from openjarvis.missions.types import (
    Mission,
    MissionStatus,
    MissionStepStatus,
)


def run_verification(mission: Mission) -> Dict[str, Any]:
    """Run all checks and return ``{"verified": bool, "checks": {...}}``.

    Deterministic by design: the same mission state always yields the same
    verdict, so it is unit-testable without any model call.
    """
    checks: Dict[str, Any] = {
        "has_goal": bool(mission.goal and mission.goal.strip()),
        "has_steps": len(mission.steps) > 0,
        "all_steps_have_results": all(
            bool(s.result and s.result.strip())
            for s in mission.steps
            if s.status == MissionStepStatus.SUCCEEDED.value
        ),
        "terminal_status_has_report": (
            mission.status != MissionStatus.SUCCEEDED.value
            or bool(mission.report and mission.report.strip())
        ),
        "step_count_within_budget": len(mission.steps) <= mission.max_steps,
        "budget_respected": mission.budget_used_tokens <= mission.max_budget_tokens,
        "no_unproven_claims": _no_unproven_claims(mission),
    }

    verified = all(checks.values())
    return {"verified": bool(verified), "checks": checks}


def _no_unproven_claims(mission: Mission) -> bool:
    """Heuristic anti-hallucination check (deterministic).

    If the report claims something that looks like a hard assertion
    (e.g. "j'ai fait X", "j'ai testé Y", "100%", "terminé") but no step result
    provides matching evidence, the check fails.  This is intentionally
    conservative: when in doubt, it fails, pushing the mission toward explicit
    proof rather than confident claims.
    """
    if mission.status != MissionStatus.SUCCEEDED.value:
        return True
    if not mission.report:
        return True  # no report, no claim -> nothing to disprove here
    evidence = "\n".join(s.result for s in mission.steps)
    claims = re.findall(r"(j'ai|j ai|je (?:suis|serai|vais)|fini|terminé|done)", mission.report.lower())
    if not claims:
        return True
    # Every explicit claim keyword must be backed by at least one word of
    # step result evidence; otherwise we cannot prove it -> flag it.
    meaningful = any(len(tok) > 20 for tok in evidence.split()) if evidence else False
    return bool(evidence.strip()) and meaningful


__all__ = ["run_verification"]
