"""MissionEngine: background, resumable mission execution.

Design (Phase 4 / D10):

* **Persistence-first.**  Every step result is checkpointed to
  :class:`~openjarvis.missions.store.MissionStore` before the next step runs.
  If the process dies mid-mission, the store holds the exact state; on restart
  :meth:`resume_inflight` re-enqueues every non-terminal mission and the
  worker continues from the last checkpoint — the "crash simulé" criterion.

* **Non-blocking.**  ``launch()`` persists the mission and returns
  immediately.  A dedicated worker thread executes steps out of band, so a
  long mission never blocks ``jarvis serve`` request handling.

* **Event-driven.**  Lifecycle changes are published on the event bus
  (``MISSION_*``), which REST/SSE subscribers and Telegram notifications
  consume.

* **BudgetGuard.**  ``max_steps`` and ``max_budget_tokens`` are enforced per
  mission.  Hitting the step cap suspends the mission instead of letting it
  burn free-tier budget (decision D9).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Iterable, List, Optional, Set

from openjarvis.core.events import EventBus, EventType
from openjarvis.missions.store import MissionStore
from openjarvis.missions.types import (
    Mission,
    MissionEvent,
    MissionStatus,
    MissionStep,
    MissionStepStatus,
)
from openjarvis.missions.verifier import run_verification

logger = logging.getLogger(__name__)

_STOP = object()

# Retry policy: transient failures are retried with exponential backoff
# before the mission is marked FAILED (spec: "retry with backoff + fallback").
_MAX_RETRIES = 3


class MissionEngine:
    """Background worker that executes missions asynchronously with resume."""

    def __init__(
        self,
        store: MissionStore,
        system: Any = None,
        *,
        event_bus: EventBus | None = None,
        notifier: Callable[[str, str, str], None] | None = None,
        poll_interval: float = 0.5,
        default_autonomy: int = 1,
        max_steps: int = 10,
        max_budget_tokens: int = 50000,
        backoff_base: float = 2.0,
        worker_capabilities: Optional[Iterable[str]] = None,
        report_base_url: str = "",
        coding_agent: Optional[Any] = None,
        heavy_agent: Optional[Any] = None,
        heavy_agents: Optional[List[Any]] = None,
    ) -> None:
        self._store = store
        self._system = system
        self._event_bus = event_bus
        # notifier(target, title, message) — e.g. wired to the Telegram bridge.
        self._notifier = notifier
        self._poll_interval = poll_interval
        self._backoff_base = backoff_base
        self._default_autonomy = default_autonomy
        self._max_steps = max_steps
        self._max_budget_tokens = max_budget_tokens
        self._worker_capabilities: Set[str] = set(worker_capabilities or [])
        self._report_base_url = report_base_url.strip().rstrip("/")
        # Phase 5: injected coding agent (e.g. OpenCodeAgent). Steps that
        # declare the ``coding`` capability dispatch here instead of a plain
        # LLM answer. Falls back to the plain system when absent.
        self._coding_agent = coding_agent
        # "Frontier" tier(s) (e.g. ClaudeSubscriptionAgent, GeminiSubscriptionAgent)
        # for steps that set MissionStep.prefer_heavy. Tried in order --
        # ``heavy_agent`` (singular, kept for backwards compatibility) goes
        # first if given, then ``heavy_agents``. Each is a *soft* preference:
        # unlike required_capabilities it never blocks the mission -- unavailable
        # or failing heavy_agent just falls back to coding_agent/system.
        agents: List[Any] = list(heavy_agents or [])
        if heavy_agent is not None and heavy_agent not in agents:
            agents.insert(0, heavy_agent)
        self._heavy_agents: List[Any] = agents
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread and resume any in-flight missions."""
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop,
            name="jarvis-mission-engine",
            daemon=True,
        )
        self._thread.start()
        resumed = self.resume_inflight()
        if resumed:
            logger.info("MissionEngine resumed %d in-flight mission(s)", len(resumed))

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to drain and stop (checkpoints are already saved)."""
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # -- public API ----------------------------------------------------------

    def launch(
        self,
        goal: str,
        *,
        steps: Optional[List[MissionStep]] = None,
        kind: str = "",
        autonomy_level: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_budget_tokens: Optional[int] = None,
        requested_by: str = "",
        metadata: Optional[dict] = None,
    ) -> Mission:
        """Persist and enqueue a new mission.  Returns immediately.

        ``kind="coding_pr"`` auto-plans the mission as 5 checkpointed steps
        (setup / implement / test / review / ship) instead of one mega-step —
        see :func:`coding_pr_steps`. ``kind="research"`` auto-plans a sourced,
        cross-checked research mission as 3 checkpointed steps — see
        :func:`research_steps`. Both ignored if ``steps`` is given explicitly.
        """
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("A mission needs a goal")
        if steps is None and kind == "coding_pr":
            steps = coding_pr_steps(goal)
        elif steps is None and kind == "research":
            steps = research_steps(goal)
        mission = Mission(
            mission_id=uuid.uuid4().hex[:16],
            goal=goal,
            created_at=time.time(),
            updated_at=time.time(),
            autonomy_level=(
                self._default_autonomy
                if autonomy_level is None
                else int(autonomy_level)
            ),
            max_steps=self._max_steps if max_steps is None else int(max_steps),
            max_budget_tokens=(
                self._max_budget_tokens
                if max_budget_tokens is None
                else int(max_budget_tokens)
            ),
            steps=steps or _default_steps(goal),
            requested_by=requested_by,
            metadata=dict(metadata or {}),
        )
        self._store.create_mission(mission)
        self._append_event(
            mission.mission_id,
            "created",
            {"goal": goal, "autonomy_level": mission.autonomy_level},
        )
        self._publish(EventType.MISSION_CREATED, {"mission_id": mission.mission_id})
        self.submit(mission.mission_id)
        return mission

    def submit(self, mission_id: str) -> bool:
        """Enqueue a mission id for processing.  Non-blocking, never raises."""
        if not self._running.is_set():
            return False
        try:
            self._queue.put_nowait(mission_id)
            return True
        except queue.Full:
            return False

    def status(self, mission_id: str) -> Optional[Mission]:
        """Return the live mission state (or ``None`` if unknown)."""
        return self._store.get_mission(mission_id)

    def pause(self, mission_id: str) -> Optional[Mission]:
        """Pause a mission between steps (autonomy level 0 gate)."""
        mission = self._store.get_mission(mission_id)
        if mission is None or mission.is_terminal:
            return mission
        if mission.status == MissionStatus.RUNNING.value:
            # The worker will notice on its next iteration.
            mission.metadata["pause_requested"] = True
        mission.status = MissionStatus.PAUSED.value
        self._store.save_mission(mission)
        self._append_event(mission_id, "paused", {})
        return mission

    def resume(self, mission_id: str) -> Optional[Mission]:
        """Resume a paused/pending mission from its checkpoint."""
        mission = self._store.get_mission(mission_id)
        if mission is None or mission.is_terminal:
            return mission
        mission.status = MissionStatus.PENDING.value
        mission.metadata.pop("pause_requested", None)
        self._store.save_mission(mission)
        self._append_event(mission_id, "resumed", {})
        self._publish(EventType.MISSION_RESUMED, {"mission_id": mission_id})
        self.submit(mission_id)
        return mission

    def cancel(self, mission_id: str) -> Optional[Mission]:
        """Cancel a non-terminal mission."""
        mission = self._store.get_mission(mission_id)
        if mission is None or mission.is_terminal:
            return mission
        mission.status = MissionStatus.CANCELLED.value
        self._store.save_mission(mission)
        self._append_event(mission_id, "cancelled", {})
        return mission

    def resume_inflight(self) -> List[Mission]:
        """Re-enqueue every non-terminal mission (crash recovery hook)."""
        inflight = self._store.list_inflight()
        for mission in inflight:
            if mission.status != MissionStatus.PAUSED.value:
                mission.status = MissionStatus.PENDING.value
                self._store.save_mission(mission)
                self._append_event(mission.mission_id, "restarted_after_crash", {})
            self.submit(mission.mission_id)
        return inflight

    # -- worker capabilities (D12) -------------------------------------------

    def register_capabilities(self, capabilities: Iterable[str]) -> List[str]:
        """Declare capabilities this worker provides; resume matching missions."""
        added = set(capabilities) - self._worker_capabilities
        self._worker_capabilities |= added
        resumed: List[str] = []
        for mission in self._store.list_inflight():
            if mission.status != MissionStatus.WAITING_FOR_WORKER.value:
                continue
            waiting = mission.metadata.get("waiting_for_worker") or {}
            missing = waiting.get("missing_capabilities", [])
            if missing and not set(missing) - self._worker_capabilities:
                mission.status = MissionStatus.PENDING.value
                self._store.save_mission(mission)
                self._append_event(mission.mission_id, "worker_online", {})
                resumed.append(mission.mission_id)
        for mission_id in resumed:
            self.submit(mission_id)
        return resumed

    def _missing_capabilities(self, step: MissionStep) -> List[str]:
        required = getattr(step, "required_capabilities", None) or []
        if not required:
            return []
        return [c for c in required if c not in self._worker_capabilities]

    # -- worker ---------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            try:
                mission_id = self._queue.get(timeout=self._poll_interval)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if mission_id is _STOP:
                self._queue.task_done()
                break
            try:
                self._execute(mission_id)
            except Exception:  # noqa: BLE001 — a bad mission must not kill the worker
                logger.exception("Mission execution failed: %s", mission_id)
            finally:
                self._queue.task_done()

    def _execute(self, mission_id: str) -> None:
        mission = self._store.get_mission(mission_id)
        if mission is None or mission.is_terminal:
            return
        if mission.status == MissionStatus.PAUSED.value:
            return
        mission.status = MissionStatus.RUNNING.value
        self._store.save_mission(mission)
        self._publish(EventType.MISSION_START, {"mission_id": mission_id})

        for step in mission.steps:
            if mission.is_terminal or mission.status != MissionStatus.RUNNING.value:
                break
            if step.status == MissionStepStatus.SUCCEEDED.value:
                continue
            if self._step_exceeded_budget(mission):
                self._suspend_for_budget(mission)
                break

            # D12: a step declares required capabilities; if the current
            # worker lacks one, the mission waits for a capable worker.
            missing = self._missing_capabilities(step)
            if missing:
                mission.status = MissionStatus.WAITING_FOR_WORKER.value
                mission.metadata["waiting_for_worker"] = {
                    "step": step.index,
                    "missing_capabilities": list(missing),
                }
                self._store.save_mission(mission)
                self._append_event(
                    mission_id,
                    "waiting_for_worker",
                    {"step": step.index, "missing": list(missing)},
                )
                return

            step.status = MissionStepStatus.RUNNING.value
            step.started_at = time.time()
            self._store.save_mission(mission)
            self._publish(
                EventType.MISSION_STEP_START,
                {"mission_id": mission_id, "step": step.index},
            )
            self._append_event(
                mission_id, "step_start", {"step": step.index, "title": step.title}
            )

            ok, result, error = self._run_step(mission, step)
            if ok:
                step.status = MissionStepStatus.SUCCEEDED.value
                step.result = result
                mission.budget_used_tokens += _estimate_tokens(result)
                self._publish(
                    EventType.MISSION_STEP_END,
                    {"mission_id": mission_id, "step": step.index},
                )
            else:
                step.status = MissionStepStatus.FAILED.value
                step.error = error
                step.finished_at = time.time()
                mission.current_step = step.index + 1
                self._store.save_mission(mission)
                self._append_event(
                    mission_id,
                    "step_failed",
                    {"step": step.index, "error": error[:500]},
                )
                self._fail(mission, f"Étape {step.index} a échoué : {error}")
                return
            step.finished_at = time.time()
            mission.current_step = step.index + 1
            # Checkpoint BEFORE the next step — crash-safe by construction.
            self._store.save_mission(mission)

            # Autonomy 0: require explicit approval between every step.
            if mission.autonomy_level == 0:
                mission.status = MissionStatus.PAUSED.value
                self._store.save_mission(mission)
                self._append_event(mission_id, "waiting_for_approval", {})
                return

        if mission.status == MissionStatus.RUNNING.value:
            self._finish(mission)

    def _run_step(self, mission: Mission, step: MissionStep):
        """Execute one step with retry/backoff.  Returns (ok, result, error)."""
        last_error = ""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._ask_step(mission, step)
                step.retries = attempt - 1
                return True, result, ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.debug(
                    "Mission %s step %d attempt %d failed: %s",
                    mission.mission_id,
                    step.index,
                    attempt,
                    last_error,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(self._backoff_base * attempt)
        return False, "", last_error

    def _ask_step(self, mission: Mission, step: MissionStep) -> str:
        """Run the step through the system (agent + tools), returning text."""
        is_coding = "coding" in (getattr(step, "required_capabilities", None) or [])
        prompt = step.prompt or mission.goal
        if is_coding:
            # Each checkpointed step gets its own fresh orchestrator turn
            # budget (see coding_pr_steps). Without this, a later step has no
            # idea earlier steps already cloned/branched, and re-does that
            # work -- the redundant rebase/re-clone loop observed in the
            # single-mega-step version of Phase 5.
            prompt = self._with_step_context(mission, step, prompt)

        heavy_already_tried = False
        if getattr(step, "prefer_heavy", False) and self._heavy_agents:
            heavy_already_tried = True
            content = self._try_heavy_agents(prompt)
            if content is not None:
                return content
            # Falls through to the normal coding_agent/system path below --
            # a soft preference must never fail a step just because every
            # frontier tier was unavailable or over budget.

        try:
            if is_coding and self._coding_agent is not None:
                result = self._coding_agent.run(prompt)
                content = getattr(result, "content", None) or ""
                if not str(content).strip():
                    raise RuntimeError("Coding step returned an empty result")
                return str(content)
            if self._system is None:
                raise RuntimeError("MissionEngine has no system to execute steps with")
            result = self._system.ask(
                prompt,
                context=False,
            )
            content = (result or {}).get("content", "")
            if not content or not str(content).strip():
                raise RuntimeError("Step returned an empty result")
            return str(content)
        except Exception:
            # Last-resort fallback for *every* step, not just prefer_heavy
            # ones: a quota-exhausted or down default provider (observed in
            # practice -- Groq's free-tier daily token cap) must not fail a
            # step outright when a frontier subscription tier could still
            # answer it. Skipped if prefer_heavy already tried the same
            # agents above (avoids a redundant, already-failing call).
            if not heavy_already_tried and self._heavy_agents:
                content = self._try_heavy_agents(prompt)
                if content is not None:
                    return content
            raise

    def _try_heavy_agents(self, prompt: str) -> Optional[str]:
        """Best-effort call across the frontier tiers, in order (e.g. Claude
        subscription, then Gemini subscription). Returns ``None`` (never
        raises) once every candidate has failed, so the caller can fall
        back to the normal coding_agent/system path cleanly. This is also
        how work distributes across subscriptions: whichever tier is
        logged in / under budget / not rate-limited answers first."""
        for agent in self._heavy_agents:
            try:
                result = agent.run(prompt)
            except Exception:  # noqa: BLE001
                logger.debug("Heavy agent %r call failed, trying next", agent, exc_info=True)
                continue
            if getattr(result, "metadata", None) and result.metadata.get("error"):
                logger.debug("Heavy agent %r returned an error, trying next: %s", agent, result.metadata)
                continue
            content = getattr(result, "content", None) or ""
            if str(content).strip():
                return str(content)
        return None

    def _with_step_context(self, mission: Mission, step: MissionStep, prompt: str) -> str:
        """Prepend a summary of already-succeeded steps to a coding prompt.

        A multi-step coding mission (setup / implement / test / review /
        ship) is dispatched as separate orchestrator calls, each starting
        with no memory of the others. Without this, "Ship" doesn't know a
        branch already exists and may try to recreate it. With it, every
        step sees what earlier steps already did (and reported), so it picks
        up from there instead of redoing it.
        """
        done = [
            s
            for s in mission.steps
            if s.index < step.index and s.status == MissionStepStatus.SUCCEEDED.value
        ]
        if not done:
            return prompt
        lines = ["Contexte : étapes déjà terminées dans cette mission (ne les refais pas) :"]
        for s in done:
            summary = " ".join((s.result or "").split())[:400]
            lines.append(f"- Étape {s.index} ({s.title}) : {summary}")
        lines.append("")
        lines.append(prompt)
        return "\n".join(lines)

    # -- completion -----------------------------------------------------------

    def _finish(self, mission: Mission) -> None:
        mission.status = MissionStatus.SUCCEEDED.value
        mission.report = _build_report(mission)
        mission.verification = run_verification(mission)
        self._store.save_mission(mission)
        self._append_event(
            mission.mission_id,
            "succeeded",
            {"verified": mission.verification.get("verified")},
        )
        self._publish(
            EventType.MISSION_END,
            {
                "mission_id": mission.mission_id,
                "verified": mission.verification.get("verified"),
            },
        )
        self._notify(mission, title="MISSION TERMINÉE")

    def _fail(self, mission: Mission, reason: str) -> None:
        mission.status = MissionStatus.FAILED.value
        mission.report = f"Mission échouée : {reason}"
        self._store.save_mission(mission)
        self._append_event(mission.mission_id, "failed", {"reason": reason})
        self._publish(
            EventType.MISSION_FAILED, {"mission_id": mission.mission_id, "reason": reason}
        )
        self._notify(mission, title="MISSION ÉCHOUÉE", message=reason)

    def _step_exceeded_budget(self, mission: Mission) -> bool:
        return (
            mission.budget_used_tokens >= mission.max_budget_tokens
            or len(mission.steps) > mission.max_steps
        )

    def _suspend_for_budget(self, mission: Mission) -> None:
        mission.status = MissionStatus.PAUSED.value
        mission.metadata["budget_exceeded"] = True
        self._store.save_mission(mission)
        self._append_event(mission.mission_id, "budget_exceeded", {})
        self._publish(
            EventType.MISSION_BUDGET_EXCEEDED, {"mission_id": mission.mission_id}
        )
        self._notify(
            mission,
            title="MISSION SUSPENDUE (budget)",
            message="Budget max atteint. Dis 'reprends la mission' pour continuer.",
        )

    # -- notifications & events ------------------------------------------------

    def _notify(
        self, mission: Mission, *, title: str, message: str = ""
    ) -> None:
        target = mission.requested_by
        if not target or self._notifier is None:
            return
        body = message or f"{mission.goal}\nID: {mission.mission_id}"
        if self._report_base_url:
            body = (
                f"{body}\nRapport : {self._report_base_url}"
                f"/v1/missions/{mission.mission_id}"
            )
        try:
            self._notifier(target, title, body)
        except Exception:  # noqa: BLE001
            logger.debug("Mission notification failed", exc_info=True)

    def _append_event(self, mission_id: str, event_type: str, data: dict) -> None:
        try:
            self._store.append_event(
                MissionEvent(
                    mission_id=mission_id,
                    ts=time.time(),
                    event_type=event_type,
                    data=data,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to append mission event", exc_info=True)

    def _publish(self, event_type: EventType, data: dict) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(event_type, data)
        except Exception:  # noqa: BLE001
            logger.debug("Mission event publish failed", exc_info=True)


# ---------------------------------------------------------------------------


def _default_steps(goal: str) -> List[MissionStep]:
    """Default plan for a mission: a single research/analysis step."""
    return [
        MissionStep(
            index=0,
            title="Analyse",
            prompt=(
                f"Mission : {goal}\n"
                "Effectue la mission. Sois factuel : si tu n'es pas sûr, dis-le. "
                "Termine par un rapport court en français : objectif, ce qui a été "
                "fait, preuves, résultats, points restants, niveau de confiance."
            ),
        )
    ]


def coding_pr_steps(goal: str) -> List[MissionStep]:
    """Checkpointed 5-phase plan for a coding mission that ends in an open PR.

    Phase 5 (2026-08-12) ran this as a single mega-step: one orchestrator
    call doing clone -> branch -> implement -> test -> review -> commit ->
    push -> PR. It worked, but did redundant git operations (repeated
    ``pull --rebase``) and hit ``max_turns`` before producing a clean final
    report. Splitting it into separate ``MissionStep``s fixes both: each
    phase gets its own fresh orchestrator turn budget instead of sharing one,
    the engine checkpoints to disk after every succeeded step (so a crash or
    a stuck phase only loses that phase, not the whole mission), and
    ``_ask_step``/``_with_step_context`` feeds each phase a summary of what
    earlier phases already did so it does not redo it.
    """
    phases = [
        (
            "Setup",
            "Étape 1/5 (Setup). Clone le dépôt cible si nécessaire, crée une "
            "branche feature dédiée (nom explicite lié à l'objectif), et "
            "positionne-toi dedans. Ne fais AUCUNE modification de code ici. "
            "Termine en indiquant clairement : chemin du dépôt local, nom de "
            "la branche créée, commit de départ.",
            False,
        ),
        (
            "Implement",
            "Étape 2/5 (Implement). Sur la branche déjà créée (voir contexte "
            "ci-dessus, ne la recrée pas), implémente le changement demandé. "
            "Reste isolé : ne touche que les fichiers nécessaires. "
            "N'exécute pas encore les tests. Termine en listant les fichiers "
            "créés/modifiés.",
            False,
        ),
        (
            "Test",
            "Étape 3/5 (Test). Exécute réellement les tests/build/lint "
            "pertinents du projet sur la branche en cours. Si un test échoue, "
            "corrige et relance jusqu'à succès (ou explique précisément "
            "pourquoi ce n'est pas possible). Termine par le résultat réel "
            "des tests (nombre passés/échoués), jamais une supposition.",
            False,
        ),
        (
            "Review",
            "Étape 4/5 (Review). Relis ton propre diff (`git diff`) comme un "
            "reviewer exigeant : régressions, fichiers non liés touchés par "
            "erreur, oublis. Corrige si besoin et re-vérifie. Termine par un "
            "résumé du diff final.",
            True,  # prefer_heavy: this is the step where catching a bug
            # is worth more than saving a few seconds -- routed to the
            # frontier tier (ClaudeSubscriptionAgent) when configured,
            # with an automatic fallback to the normal coding agent.
        ),
        (
            "Ship",
            "Étape 5/5 (Ship). Commit (message clair), push la branche déjà "
            "créée, puis ouvre une pull request (`gh pr create`) vers la "
            "branche par défaut. Ne refais PAS le clone ni la branche (déjà "
            "faits à l'étape 1) ni un rebase inutile. Termine par l'URL de la "
            "PR ouverte.",
            False,
        ),
    ]
    return [
        MissionStep(
            index=i,
            title=title,
            prompt=f"Mission : {goal}\n\n{body}",
            required_capabilities=["coding"],
            prefer_heavy=prefer_heavy,
        )
        for i, (title, body, prefer_heavy) in enumerate(phases)
    ]


def research_steps(goal: str) -> List[MissionStep]:
    """Checkpointed 3-phase plan for a sourced, cross-checked research mission
    (spec §5/§26/§34: multi-hop, cross-check, contradictions, primary
    sources, argued conclusion). Same rationale as ``coding_pr_steps``: each
    phase is a separate orchestrator call with its own turn budget and its
    own checkpoint, instead of one call expected to search, verify, and
    write up in a single pass.
    """
    phases = [
        (
            "Recherche",
            "Étape 1/3 (Recherche). Utilise `web_search`/`knowledge_search` "
            "pour rassembler des sources concrètes sur le sujet (plusieurs "
            "requêtes si besoin, plusieurs sources). Pour chaque élément "
            "trouvé, note la source (titre/URL). Ne conclus rien encore. "
            "Termine en listant les sources trouvées et ce que chacune dit.",
            False,
        ),
        (
            "Vérification croisée",
            "Étape 2/3 (Vérification croisée). Sur les sources de l'étape "
            "précédente : identifie les faits confirmés par plusieurs "
            "sources indépendantes, les contradictions entre sources, et ce "
            "qui n'est qu'une opinion/non confirmé. Distingue sources "
            "primaires et secondaires. Termine par la liste des points "
            "solides et des points incertains/contradictoires.",
            False,
        ),
        (
            "Synthèse",
            "Étape 3/3 (Synthèse). Rédige la réponse finale argumentée : "
            "conclusion claire, recommandation si pertinent, sources citées, "
            "incertitudes/contradictions signalées explicitement plutôt que "
            "lissées. Ne jamais affirmer plus que ce que les sources "
            "soutiennent réellement.",
            True,  # prefer_heavy: synthesizing conflicting sources into a
            # calibrated, well-argued answer benefits from the frontier
            # tier the same way code review does.
        ),
    ]
    return [
        MissionStep(
            index=i,
            title=title,
            prompt=f"Mission : {goal}\n\n{body}",
            prefer_heavy=prefer_heavy,
        )
        for i, (title, body, prefer_heavy) in enumerate(phases)
    ]


def _build_report(mission: Mission) -> str:
    lines = [
        f"# Rapport de mission — {mission.mission_id}",
        f"Objectif : {mission.goal}",
        f"Statut : TERMINÉE ({len(mission.steps)} étape(s))",
        "",
    ]
    for step in mission.steps:
        status = "OK" if step.status == MissionStepStatus.SUCCEEDED.value else "FAIL"
        lines.append(f"## Étape {step.index} [{status}] — {step.title}")
        if step.result:
            lines.append(step.result)
        if step.error:
            lines.append(f"Erreur : {step.error}")
        lines.append("")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars/token) for BudgetGuard accounting."""
    return max(1, len(text or "") // 4)


__all__ = [
    "MissionEngine",
    "_default_steps",
    "coding_pr_steps",
    "research_steps",
    "_build_report",
    "_estimate_tokens",
]
