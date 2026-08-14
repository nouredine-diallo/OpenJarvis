"""Adaptive quality gates: proportional proof before a mission is DONE.

Brique 5 (docs/SPEC_BRIQUE5_QUALITY_GATE.md). ``verifier.run_verification``
already gates every mission, but only *structurally* -- it checks that
steps produced results and that the report doesn't over-claim. It has no
idea whether a coding mission's code actually lints, passes security
checks, or has green tests. "Les tests sont verts" was still just
something a model could write.

This module adds gates that depend on what the mission actually was:

    code           -> ruff + bandit + pytest, really executed in a sandbox
    research       -> claims must carry citations and list sources
    ui             -> a visual artefact must exist
    conversational -> light gate only (structural checks)

Two principles carried from the rest of the project:

* **Rigueur proportionnelle** (decision D4): a conversational answer must
  not trigger a 30-second test run. The gate level is chosen from the
  mission's own declared capabilities/kind, and defaults to light.
* **Sandboxed, never on the host** (decision D3): checks run in the same
  throwaway-container posture as the benchmark tool (network cut, memory
  and CPU capped, timeout), and ``__pycache__`` is excluded from the copy
  so a stale cache can never manufacture green tests.

Every gate returns evidence (the raw tool output), not just a verdict --
a passing gate with no evidence would be exactly the kind of unproven
claim this project refuses to accept.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GATE_LIGHT = "light"
GATE_FULL = "full"

KIND_CODE = "code"
KIND_RESEARCH = "research"
KIND_UI = "ui"
KIND_CONVERSATIONAL = "conversational"

SANDBOX_IMAGE = "python:3.12-slim"
SANDBOX_TIMEOUT_S = 180
MEMORY_LIMIT = "512m"
CPU_LIMIT = "1"


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str = ""
    evidence: str = ""


@dataclass(slots=True)
class GateResult:
    kind: str = KIND_CONVERSATIONAL
    level: str = GATE_LIGHT
    checks: List[GateCheck] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "level": self.level,
            "passed": self.passed,
            "skipped_reason": self.skipped_reason,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in self.checks},
        }

    def as_markdown(self) -> str:
        if self.skipped_reason:
            return f"Quality gate ({self.kind}) : ignoré — {self.skipped_reason}"
        lines = [f"Quality gate ({self.kind}, niveau {self.level}) :"]
        for c in self.checks:
            lines.append(f"- {'✅' if c.passed else '❌'} **{c.name}** — {c.detail}")
        return "\n".join(lines)


def classify_mission(mission: Any) -> str:
    """Decide which gate family applies, from what the mission declared.

    Uses the steps' ``required_capabilities`` rather than guessing from
    free text: a mission that asked for the ``coding`` capability is a
    coding mission by construction, which is far more reliable than
    pattern-matching a goal written in natural language.
    """
    caps = set()
    for step in getattr(mission, "steps", []) or []:
        caps.update(getattr(step, "required_capabilities", None) or [])

    if "coding" in caps or "tests" in caps:
        return KIND_CODE
    if "preview" in caps or "browser" in caps:
        return KIND_UI

    goal = (getattr(mission, "goal", "") or "").lower()
    if any(w in goal for w in ("recherche", "cherche", "compare", "état de l'art", "sources")):
        return KIND_RESEARCH
    return KIND_CONVERSATIONAL


def gate_level_for(kind: str) -> str:
    """Full gate for code/research/UI, light for conversation (decision D4)."""
    return GATE_LIGHT if kind == KIND_CONVERSATIONAL else GATE_FULL


# -- sandboxed code checks ---------------------------------------------------


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _copy_project(src: Path, dest: Path) -> None:
    """Copy a project into a scratch dir, excluding caches and venvs.

    ``__pycache__`` is excluded deliberately: a stale cache can make tests
    appear to pass against code that no longer exists (spec §4.3 point 4).
    """
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".venv", "venv", ".git", "node_modules", ".pytest_cache"
        ),
        dirs_exist_ok=True,
    )


def run_code_checks(
    project_dir: str,
    *,
    timeout_s: int = SANDBOX_TIMEOUT_S,
    image: str = SANDBOX_IMAGE,
) -> List[GateCheck]:
    """Run ruff, bandit and pytest against *project_dir* in a container.

    Returns one :class:`GateCheck` per tool, each carrying the tool's real
    output as evidence. A missing/unavailable sandbox yields a *failing*
    check rather than a silently skipped one -- "we could not verify" must
    never read as "verified".
    """
    project = Path(project_dir)
    if not project.is_dir():
        return [GateCheck("sandbox", False, f"projet introuvable : {project_dir}")]
    if not _docker_available():
        return [
            GateCheck(
                "sandbox",
                False,
                "Docker indisponible — impossible de vérifier le code sans bac à sable "
                "(exécuter du code non vérifié sur la machine hôte est exclu).",
            )
        ]

    with tempfile.TemporaryDirectory(prefix="jarvis-gate-") as tmp:
        work = Path(tmp) / "project"
        try:
            _copy_project(project, work)
        except Exception as exc:  # noqa: BLE001
            return [GateCheck("sandbox", False, f"copie du projet impossible : {exc}")]

        # One container installs the linters once, then runs all three
        # checks -- cheaper and more predictable than three cold starts.
        script = (
            "set +e\n"
            "pip install --quiet --disable-pip-version-check ruff bandit pytest >/dev/null 2>&1\n"
            "echo '===RUFF==='\n"
            # --no-cache: the project is mounted read-only, and ruff
            # otherwise dies trying to create .ruff_cache there --
            # an infrastructure failure that would read as "lint failed".
            "ruff check --no-cache . 2>&1; echo \"EXIT:$?\"\n"
            "echo '===BANDIT==='\n"
            # -s B101: bandit flags every `assert` as a vulnerability,
            # including the ones inside test files, where asserts are
            # the entire point. Left on, the gate would fail every
            # project that has tests -- the textbook false positive the
            # spec warns against ("pas de faux négatifs abusifs").
            "bandit -q -r -s B101 . 2>&1; echo \"EXIT:$?\"\n"
            "echo '===PYTEST==='\n"
            "pytest -q 2>&1; echo \"EXIT:$?\"\n"
        )
        args = [
            "docker", "run", "--rm",
            "--name", f"jarvis-gate-{uuid.uuid4().hex[:10]}",
            "--label", "openjarvis-sandbox=true",
            "--memory", MEMORY_LIMIT,
            # Pinned to --memory: without it Docker allows swap spillover and
            # the cap is decorative (verified live -- see tools/experiment.py).
            "--memory-swap", MEMORY_LIMIT,
            "--cpus", CPU_LIMIT,
            "--pids-limit", "256",
            "-v", f"{work}:/app:ro",
            "-w", "/app",
            image, "bash", "-c", script,
        ]
        # Note: the network stays ON here (unlike the benchmark sandbox)
        # only because pip must fetch the linters. The mount is read-only,
        # so the project itself cannot be modified from inside.
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [GateCheck("sandbox", False, f"délai dépassé ({timeout_s}s)")]
        except Exception as exc:  # noqa: BLE001
            return [GateCheck("sandbox", False, f"exécution impossible : {exc}")]

    return _parse_check_output(proc.stdout or "")


def _parse_check_output(output: str) -> List[GateCheck]:
    sections = {"RUFF": "", "BANDIT": "", "PYTEST": ""}
    current = None
    for line in output.splitlines():
        marker = re.match(r"^===(RUFF|BANDIT|PYTEST)===$", line.strip())
        if marker:
            current = marker.group(1)
            continue
        if current:
            sections[current] += line + "\n"

    checks: List[GateCheck] = []
    labels = {
        "RUFF": ("lint (ruff)", "style/erreurs statiques"),
        "BANDIT": ("sécurité (bandit)", "vulnérabilités courantes"),
        "PYTEST": ("tests (pytest)", "suite de tests"),
    }
    for key, (name, what) in labels.items():
        body = sections[key]
        exit_match = re.search(r"EXIT:(\d+)\s*$", body.strip())
        if not exit_match:
            checks.append(GateCheck(name, False, f"{what} : sortie illisible", body[-500:]))
            continue
        code = int(exit_match.group(1))
        evidence = body[: body.rfind("EXIT:")].strip()
        if key == "PYTEST" and code == 5:
            # pytest exit 5 == "no tests collected". Not a failure of the
            # code, but not evidence of correctness either -- reported as a
            # failed check so "no tests" can never read as "tests passed".
            checks.append(
                GateCheck(name, False, "aucun test trouvé — impossible de prouver que ça marche", evidence[-1500:])
            )
            continue
        passed = code == 0
        detail = f"{what} : {'OK' if passed else 'échec'}"
        checks.append(GateCheck(name, passed, detail, evidence[-1500:]))
    return checks


# -- research gate -----------------------------------------------------------

_CITATION_RE = re.compile(r"\[\d+\]")


def run_research_checks(mission: Any) -> List[GateCheck]:
    """A research answer must cite, and its citations must resolve to
    listed sources (Brique 3 produces exactly this shape)."""
    text = "\n".join((getattr(s, "result", "") or "") for s in getattr(mission, "steps", []) or [])
    report = getattr(mission, "report", "") or ""
    haystack = f"{text}\n{report}"

    citations = set(_CITATION_RE.findall(haystack))
    has_sources = bool(re.search(r"(##\s*Sources|Sources?\s*:)", haystack, re.IGNORECASE))

    return [
        GateCheck(
            "citations",
            bool(citations),
            f"{len(citations)} référence(s) [n] trouvée(s)" if citations
            else "aucune citation [n] — une synthèse non sourcée n'est pas vérifiable",
        ),
        GateCheck(
            "liste de sources",
            has_sources,
            "liste de sources présente" if has_sources else "aucune liste de sources",
        ),
    ]


# -- ui gate -----------------------------------------------------------------


def run_ui_checks(mission: Any) -> List[GateCheck]:
    """A UI mission must leave a visual artefact behind."""
    artefacts = list(getattr(mission, "artefacts", None) or [])
    visual = [a for a in artefacts if str(a).lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    return [
        GateCheck(
            "preuve visuelle",
            bool(visual),
            f"{len(visual)} capture(s)" if visual else "aucune capture — le rendu n'est pas prouvé",
        )
    ]


# -- entry point -------------------------------------------------------------


def run_quality_gate(
    mission: Any,
    *,
    project_dir: str = "",
    enable_sandbox_checks: bool = True,
) -> GateResult:
    """Run the gate appropriate to *mission*. Never raises."""
    kind = classify_mission(mission)
    result = GateResult(kind=kind, level=gate_level_for(kind))

    try:
        if kind == KIND_CODE:
            if not enable_sandbox_checks:
                result.skipped_reason = "vérifications sandbox désactivées par configuration"
            elif not project_dir:
                result.skipped_reason = "aucun répertoire de projet fourni"
            else:
                result.checks = run_code_checks(project_dir)
        elif kind == KIND_RESEARCH:
            result.checks = run_research_checks(mission)
        elif kind == KIND_UI:
            result.checks = run_ui_checks(mission)
        # conversational: structural checks only, handled by run_verification
    except Exception as exc:  # noqa: BLE001
        logger.debug("Quality gate failed", exc_info=True)
        result.checks = [GateCheck("quality_gate", False, f"erreur interne : {exc}")]

    return result


__all__ = [
    "GATE_FULL",
    "GATE_LIGHT",
    "KIND_CODE",
    "KIND_CONVERSATIONAL",
    "KIND_RESEARCH",
    "KIND_UI",
    "GateCheck",
    "GateResult",
    "classify_mission",
    "gate_level_for",
    "run_code_checks",
    "run_quality_gate",
    "run_research_checks",
    "run_ui_checks",
]
