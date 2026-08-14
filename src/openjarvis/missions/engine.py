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
from pathlib import Path
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


class _AgentResultLike:
    """Minimal stand-in for openjarvis.agents._stubs.AgentResult -- just
    enough shape (.content, .metadata) for _try_agents to read."""

    __slots__ = ("content", "metadata")

    def __init__(self, content: str = "", metadata: Optional[dict] = None) -> None:
        self.content = content
        self.metadata = metadata or {}


class SystemAskAgent:
    """Adapts a plain ``JarvisSystem.ask()`` call to the ``.run(prompt) ->
    result-with-.content`` interface :meth:`MissionEngine._try_agents`
    expects -- lets a sibling free-tier system (e.g. another Groq model,
    its own separate daily-quota pool) plug into the same fallback
    machinery as the paid frontier tiers, at zero marginal cost. See
    ``default_fallback_agents`` on :class:`MissionEngine`: tried before
    ever spending Claude/Gemini subscription budget.
    """

    def __init__(self, system: Any, label: str = "") -> None:
        self._system = system
        self.label = label or getattr(system, "model", "") or "system"

    def run(self, prompt: str) -> _AgentResultLike:
        try:
            result = self._system.ask(prompt, context=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SystemAskAgent(%s) failed: %s", self.label, exc, exc_info=True)
            return _AgentResultLike(metadata={"error": True, "detail": str(exc)})
        content = (result or {}).get("content", "")
        if not content or not str(content).strip():
            return _AgentResultLike(metadata={"error": True, "detail": "empty result"})
        return _AgentResultLike(content=str(content))

    def __repr__(self) -> str:
        return f"SystemAskAgent({self.label})"


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
        default_fallback_agents: Optional[List[Any]] = None,
        photo_sender: Optional[Callable[[str, str, str], bool]] = None,
        enable_visual_proof: bool = True,
        visual_proof_min_ram_mb: float = 1024.0,
        enable_quality_gate: bool = True,
        enable_gate_sandbox: bool = True,
        quality_gate_project_dir: str = "",
    ) -> None:
        self._store = store
        self._system = system
        self._event_bus = event_bus
        # notifier(target, title, message) — e.g. wired to the Telegram bridge.
        self._notifier = notifier
        # photo_sender(target, path, caption) -> bool -- e.g. Telegram's
        # send_photo. Visual proof pipeline (2026-08-13): after a coding
        # mission succeeds, best-effort screenshot of whatever dev server
        # it left running, delivered as a photo. Silent no-op if no
        # server is detected, RAM is tight, or this isn't wired.
        self._photo_sender = photo_sender
        self._enable_visual_proof = enable_visual_proof
        self._visual_proof_min_ram_mb = visual_proof_min_ram_mb
        # Brique 5 adaptive quality gate (see missions/quality_gate.py).
        self._enable_quality_gate = enable_quality_gate
        self._enable_gate_sandbox = enable_gate_sandbox
        self._quality_gate_project_dir = quality_gate_project_dir
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
        # Free, zero-infra fallback tier (e.g. sibling Groq models --
        # separate daily-quota pools from the default one) tried when the
        # default provider fails, BEFORE ever spending Claude/Gemini
        # subscription budget. Deliberately a separate list from
        # heavy_agents: a prefer_heavy step wants quality first (skip
        # straight to Claude/Gemini), not "whichever free model answers".
        self._default_fallback_agents: List[Any] = list(default_fallback_agents or [])
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
        :func:`research_steps`. ``kind="improve"`` auto-plans an open-ended
        "analyze then propose" mission that pauses
        (``WAITING_FOR_CHOICE``) for the user to pick an option before
        anything is executed — see :func:`improve_steps` and
        :meth:`choose`. All ignored if ``steps`` is given explicitly.
        """
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("A mission needs a goal")
        if steps is None and kind == "coding_pr":
            steps = coding_pr_steps(goal)
        elif steps is None and kind == "research":
            steps = research_steps(goal)
        elif steps is None and kind == "improve":
            steps = improve_steps(goal)
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

    def choose(
        self,
        mission_id: Optional[str],
        choice: str,
        *,
        requested_by: str = "",
    ) -> Optional[Mission]:
        """Resume a WAITING_FOR_CHOICE mission with the user's free-text pick.

        Appends :func:`coding_pr_steps` for ``{original goal} + chosen
        option`` to the mission's step list and resumes -- the mission
        keeps its history (Analyse/Proposition results stay visible) rather
        than starting a new one, so ``mission_status`` still reads as one
        continuous story.

        ``mission_id=None`` picks the most recent WAITING_FOR_CHOICE
        mission (filtered to ``requested_by`` when given) so a Telegram
        reply doesn't need to quote the mission_id back -- there is
        normally at most one pending proposal per user at a time.
        """
        mission = self._resolve_choice_target(mission_id, requested_by)
        if mission is None:
            return None
        if mission.status != MissionStatus.WAITING_FOR_CHOICE.value:
            return mission

        chosen_goal = f"{mission.goal}\n\nOption choisie par l'utilisateur : {choice}"
        next_index = len(mission.steps)
        appended = coding_pr_steps(chosen_goal)
        for offset, step in enumerate(appended):
            step.index = next_index + offset
        mission.steps.extend(appended)
        mission.metadata["chosen_option"] = choice
        mission.status = MissionStatus.PENDING.value
        self._store.save_mission(mission)
        self._append_event(mission.mission_id, "choice_made", {"choice": choice})
        self._publish(EventType.MISSION_RESUMED, {"mission_id": mission.mission_id})
        self.submit(mission.mission_id)
        return mission

    def _resolve_choice_target(
        self, mission_id: Optional[str], requested_by: str
    ) -> Optional[Mission]:
        if mission_id:
            return self._store.get_mission(mission_id)
        candidates = self._store.list_missions(
            status=MissionStatus.WAITING_FOR_CHOICE.value, limit=20
        )
        if requested_by:
            candidates = [m for m in candidates if m.requested_by == requested_by]
        return candidates[0] if candidates else None

    def give_feedback(
        self,
        mission_id: Optional[str],
        feedback: str,
        *,
        requested_by: str = "",
    ) -> Optional[Mission]:
        """Spec §32: "the button is too big" -> a revision round on the
        SAME mission, not a brand new one from scratch. Only applies to a
        mission that already SUCCEEDED (a live coding round is a
        different concern -- use ``choose``/normal execution for that) and
        only if it had a coding capability (feedback on a plain research
        answer doesn't fit this shape). Appends a fresh
        :func:`coding_pr_steps` round for ``{goal} + feedback`` and
        resumes -- history (previous rounds' results) stays visible.

        ``mission_id=None`` picks the most recently SUCCEEDED mission for
        ``requested_by``, mirroring :meth:`choose` -- a reply like "le
        bouton est trop gros" doesn't need to quote a mission_id either.
        """
        mission = self._resolve_feedback_target(mission_id, requested_by)
        if mission is None:
            return None
        if mission.status != MissionStatus.SUCCEEDED.value:
            return mission
        is_coding_mission = any(
            "coding" in (getattr(s, "required_capabilities", None) or [])
            for s in mission.steps
        )
        if not is_coding_mission:
            return mission

        revised_goal = (
            f"{mission.goal}\n\nRetour de l'utilisateur après la version "
            f"précédente : {feedback}\n\nAjuste le travail déjà fait en "
            f"conséquence (ne recommence pas depuis zéro)."
        )
        next_index = len(mission.steps)
        appended = coding_pr_steps(revised_goal)
        for offset, step in enumerate(appended):
            step.index = next_index + offset
        mission.steps.extend(appended)
        mission.metadata.setdefault("feedback_rounds", []).append(feedback)
        mission.status = MissionStatus.PENDING.value
        # This mission already went through _finish() once; its prior
        # report/verification describe the OLD state and must not be
        # mistaken for the current one until the new round finishes.
        mission.report = ""
        mission.verification = {}
        self._store.save_mission(mission)
        self._append_event(mission.mission_id, "feedback_received", {"feedback": feedback})
        self._publish(EventType.MISSION_RESUMED, {"mission_id": mission.mission_id})
        self.submit(mission.mission_id)
        return mission

    def _resolve_feedback_target(
        self, mission_id: Optional[str], requested_by: str
    ) -> Optional[Mission]:
        if mission_id:
            return self._store.get_mission(mission_id)
        candidates = self._store.list_missions(
            status=MissionStatus.SUCCEEDED.value, limit=20
        )
        if requested_by:
            candidates = [m for m in candidates if m.requested_by == requested_by]
        return candidates[0] if candidates else None

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

            # spec §27: a step that just proposed options waits for the
            # user's free-text pick (MissionEngine.choose()) rather than
            # continuing on its own — independent of autonomy_level, since
            # this isn't "approve the next step", it's "which option even
            # is the next step".
            if getattr(step, "pause_for_choice", False):
                mission.status = MissionStatus.WAITING_FOR_CHOICE.value
                self._store.save_mission(mission)
                self._append_event(
                    mission_id, "waiting_for_choice", {"step": step.index}
                )
                self._notify(
                    mission,
                    title="JARVIS PROPOSE",
                    message=(
                        f"{step.result}\n\n(Réponds avec ton choix — "
                        f"mission {mission_id})"
                    ),
                )
                return

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
            content = self._try_agents(self._heavy_agents, prompt)
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
            # The default provider failed (observed in practice -- Groq's
            # free-tier daily token cap). Two-tier fallback, cheapest first:
            # 1) free/zero-infra siblings (e.g. other Groq models, their own
            #    separate quota pools) -- costs nothing, so try these before
            #    ever spending subscription budget.
            if self._default_fallback_agents:
                content = self._try_agents(self._default_fallback_agents, prompt)
                if content is not None:
                    return content
            # 2) the frontier tier, as an absolute last resort. Skipped if
            #    prefer_heavy already tried the same agents above (avoids a
            #    redundant, already-failing call).
            if not heavy_already_tried and self._heavy_agents:
                content = self._try_agents(self._heavy_agents, prompt)
                if content is not None:
                    return content
            raise

    def _try_agents(self, agents: List[Any], prompt: str) -> Optional[str]:
        """Best-effort call across a list of candidate agents, in order.
        Returns ``None`` (never raises) once every candidate has failed, so
        the caller can fall through to the next tier cleanly. This is also
        how work distributes across several free models/subscriptions:
        whichever candidate is available/under budget/not rate-limited
        answers first."""
        for agent in agents:
            try:
                result = agent.run(prompt)
            except Exception:  # noqa: BLE001
                logger.debug("Agent %r call failed, trying next", agent, exc_info=True)
                continue
            if getattr(result, "metadata", None) and result.metadata.get("error"):
                logger.debug("Agent %r returned an error, trying next: %s", agent, result.metadata)
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
        # Provisionally SUCCEEDED so run_verification's "terminal_status_has_
        # report" check evaluates correctly; reverted below if verification
        # fails. Found live (2026-08-12): every coding_pr_steps phase can
        # honestly report being blocked/refused, get checkpointed
        # "succeeded" anyway (non-empty text is not proof of real work),
        # and the mission used to report MISSION TERMINÉE regardless --
        # verification existed but was purely advisory metadata, never a
        # gate on the actual status. It gates now.
        mission.status = MissionStatus.SUCCEEDED.value
        mission.report = _build_report(mission)
        mission.verification = run_verification(mission)
        verified = bool(mission.verification.get("verified"))

        # Brique 5: on top of the structural checks above, run the gate
        # appropriate to what this mission actually was -- for code that
        # means really executing ruff/bandit/pytest in a sandbox, so
        # "les tests sont verts" stops being something a model can simply
        # assert. Failing the gate fails the mission, exactly like the
        # structural verification.
        gate = self._run_quality_gate(mission)
        if gate is not None:
            mission.verification["quality_gate"] = gate.as_dict()
            if not gate.passed:
                verified = False

        # An explicit human override can still let a mission through, but
        # only when a reason was recorded (via give_feedback/choose) --
        # never silently, and never by the model itself.
        override = (mission.metadata or {}).get("override_reason", "")
        if not verified and override:
            verified = True
            mission.verification["overridden_by_human"] = override

        if not verified:
            mission.status = MissionStatus.FAILED.value
            self._store.save_mission(mission)
            self._append_event(
                mission.mission_id,
                "failed_verification",
                {"checks": mission.verification.get("checks")},
            )
            self._publish(
                EventType.MISSION_FAILED,
                {"mission_id": mission.mission_id, "verified": False},
            )
            self._notify(
                mission,
                title="MISSION NON VÉRIFIÉE",
                message=(
                    "Toutes les étapes ont répondu, mais la vérification "
                    "anti-hallucination a échoué (au moins une étape "
                    "rapporte un blocage réel ou une preuve manquante) -- "
                    "voir le rapport avant de considérer ceci comme fait."
                ),
            )
            return

        # Visual proof BEFORE the checkpoint save (not after): _finish()
        # used to save, then run this, then save again -- a stale-write
        # race found live via a flaky test: give_feedback() (or choose(),
        # cancel(), any other concurrent mutator) reading+modifying+saving
        # the mission during that gap had its write silently clobbered by
        # this method's own second save, which was still holding the
        # PRE-visual-proof in-memory copy. One save at the end, after every
        # mutation (including the artefact append), closes that window.
        captured_path = self._try_visual_proof(mission)
        mission.report = _build_report(
            mission, report_base_url=self._report_base_url
        )
        self._store.save_mission(mission)
        self._append_event(
            mission.mission_id,
            "succeeded",
            {"verified": True},
        )
        self._publish(
            EventType.MISSION_END,
            {"mission_id": mission.mission_id, "verified": True},
        )
        self._notify(mission, title="MISSION TERMINÉE")
        if captured_path:
            # Spawned only *after* the checkpoint save above has landed,
            # not from inside _try_visual_proof: the backup worker re-reads
            # the mission from the store and saves its own patch, so
            # starting it earlier would race this method's own save and
            # risk being silently clobbered by it (same class of bug as
            # the stale-write race documented above).
            self._try_backup_artifact(mission, captured_path)

    def _try_visual_proof(self, mission: Mission) -> Optional[str]:
        """Best-effort screenshot of whatever dev server a coding mission
        left running, sent as a photo. Never raises, never delays/blocks
        the mission's own completion -- every failure mode (disabled,
        no photo_sender wired, no server detected, RAM too tight,
        playwright missing, capture error) is a silent no-op, logged at
        debug level and recorded as a mission event for auditability.
        Returns the captured artefact's local path on success, else None."""
        if not self._enable_visual_proof or self._photo_sender is None:
            return None
        is_coding_mission = any(
            "coding" in (getattr(s, "required_capabilities", None) or [])
            for s in mission.steps
        )
        if not is_coding_mission:
            return None
        try:
            from openjarvis.tools.screenshot import capture_screenshot, find_dev_server_url

            evidence = "\n".join(s.result for s in mission.steps)
            url = find_dev_server_url(evidence)
            if not url:
                self._append_event(
                    mission.mission_id, "visual_proof_skipped", {"reason": "no_server_detected"}
                )
                return None
            out_path = str(
                Path.home() / ".openjarvis" / "mission_artifacts" / mission.mission_id / "final.png"
            )
            path, reason = capture_screenshot(
                url, out_path, min_ram_mb=self._visual_proof_min_ram_mb
            )
            if not path:
                self._append_event(
                    mission.mission_id, "visual_proof_skipped", {"reason": reason}
                )
                return None
            mission.artefacts.append(path)
            # No save here -- the caller (_finish) does exactly one save
            # after every mutation (including this one) to avoid the
            # stale-write race a double-save used to cause (see _finish).
            sent = self._photo_sender(
                mission.requested_by, path, f"Résultat — {mission.goal[:200]}"
            )
            self._append_event(
                mission.mission_id,
                "visual_proof_captured",
                {"url": url, "path": path, "sent": bool(sent)},
            )
            return path
        except Exception:  # noqa: BLE001
            logger.debug("Visual proof capture failed", exc_info=True)
            self._append_event(mission.mission_id, "visual_proof_skipped", {"reason": "error"})

    def _run_quality_gate(self, mission: Mission):
        """Run the adaptive quality gate. Never raises -- a gate that
        crashes must not take the mission down with it, but it also must
        not silently pass: an internal error surfaces as a failed check
        (see quality_gate.run_quality_gate)."""
        if not self._enable_quality_gate:
            return None
        try:
            from openjarvis.missions.quality_gate import run_quality_gate

            return run_quality_gate(
                mission,
                project_dir=(mission.metadata or {}).get("workspace", "")
                or self._quality_gate_project_dir,
                enable_sandbox_checks=self._enable_gate_sandbox,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Quality gate execution failed", exc_info=True)
            return None

    def _try_backup_artifact(self, mission: Mission, local_path: str) -> None:
        """Fire-and-forget: push the artefact to the permanent GitHub
        backup repo in a background thread so a slow or unreachable
        network never delays the mission's own completion/notification --
        the git clone/push round-trip can take several seconds on a first
        run, far more than a mission's own checkpoint save should ever
        wait on. The mission is re-read from the store and patched once
        the push finishes (which is typically after the mission has
        already been reported as done); _build_report simply keeps using
        the local artifacts API URL for this artefact until then."""

        def _worker() -> None:
            try:
                from openjarvis.tools.artifact_backup import push_artifact

                url = push_artifact(local_path, mission.mission_id)
                if not url:
                    return
                fresh = self._store.get_mission(mission.mission_id)
                if fresh is None:
                    return
                # This can race _finish()'s own checkpoint save (this thread
                # is spawned from inside _try_visual_proof, before _finish
                # persists mission.artefacts.append(local_path)) -- if we
                # read the store before that save lands, local_path won't be
                # in fresh.artefacts yet. Self-heal rather than lose the URL.
                if local_path not in fresh.artefacts:
                    fresh.artefacts.append(local_path)
                urls = fresh.metadata.setdefault("artefact_github_urls", {})
                urls[local_path] = url
                fresh.report = _build_report(fresh, report_base_url=self._report_base_url)
                self._store.save_mission(fresh)
                self._append_event(
                    mission.mission_id, "artifact_backed_up", {"path": local_path, "url": url}
                )
            except Exception:  # noqa: BLE001
                logger.debug("Artifact GitHub backup failed", exc_info=True)

        threading.Thread(target=_worker, daemon=True, name="artifact-backup").start()

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


def improve_steps(goal: str) -> List[MissionStep]:
    """Checkpointed plan for spec §27 -- open-ended requests ("look at my
    app and improve it") get analyzed and proposed *before* anything is
    touched, unlike :func:`coding_pr_steps` which assumes the goal is
    already a concrete instruction ("add X").

    Two steps: Analyse, then Proposition. Proposition sets
    ``pause_for_choice=True`` -- the engine stops there
    (``MissionStatus.WAITING_FOR_CHOICE``) instead of executing anything,
    and :meth:`MissionEngine.choose` appends the concrete
    :func:`coding_pr_steps` for whichever option the user picks.
    """
    return [
        MissionStep(
            index=0,
            title="Analyse",
            prompt=(
                f"Mission : {goal}\n\n"
                "Étape 1/2 (Analyse). Inspecte réellement le projet/dépôt "
                "concerné : structure, conventions, technologies, ce qui "
                "existe déjà. Cite les fichiers que tu as vraiment lus. Ne "
                "propose rien encore, ne modifie rien. Termine par un "
                "résumé factuel de ce que tu as trouvé."
            ),
            required_capabilities=["coding"],
        ),
        MissionStep(
            index=1,
            title="Proposition",
            prompt=(
                f"Mission : {goal}\n\n"
                "Étape 2/2 (Proposition). Sur la base de l'analyse "
                "ci-dessus (contexte fourni juste avant ce message), "
                "propose 3 à 4 améliorations concrètes distinctes. Pour "
                "chacune : ce que ça change, pourquoi c'est utile, une "
                "priorité justifiée. Numérote-les clairement (1., 2., "
                "3., ...). Ne code rien, n'ouvre aucun fichier en "
                "écriture. Termine en demandant explicitement laquelle "
                "implémenter (un numéro, plusieurs, ou une autre idée)."
            ),
            required_capabilities=["coding"],
            pause_for_choice=True,
        ),
    ]


def _confidence_label(mission: Mission) -> str:
    """Spec §33: a confidence level, not just "done"/"not done". Derived
    from the verification checks that already ran (deterministic, not a
    model's own self-assessment) -- "Élevée" only when every check the
    anti-hallucination gate ran actually passed clean."""
    checks = (mission.verification or {}).get("checks") or {}
    if not checks:
        return "Non évaluée"
    if all(checks.values()):
        return "Élevée (toutes les vérifications passées)"
    failed = [name for name, ok in checks.items() if not ok]
    return f"Faible ({', '.join(failed)})"


def _build_report(mission: Mission, *, report_base_url: str = "") -> str:
    """Spec §31/§33: objectif, étapes, preuves (diff/tests/capture),
    niveau de confiance, points restants -- not just a step-by-step log.
    """
    lines = [
        f"# Rapport de mission — {mission.mission_id}",
        f"Objectif : {mission.goal}",
        f"Statut : TERMINÉE ({len(mission.steps)} étape(s))",
        f"Niveau de confiance : {_confidence_label(mission)}",
    ]
    rounds = (mission.metadata or {}).get("feedback_rounds") or []
    if rounds:
        lines.append(f"Retours pris en compte : {len(rounds)} — {'; '.join(rounds)}")
    lines.append("")

    for step in mission.steps:
        status = "OK" if step.status == MissionStepStatus.SUCCEEDED.value else "FAIL"
        lines.append(f"## Étape {step.index} [{status}] — {step.title}")
        if step.result:
            lines.append(step.result)
        if step.error:
            lines.append(f"Erreur : {step.error}")
        lines.append("")

    if mission.artefacts:
        lines.append("## Preuves")
        github_urls = (mission.metadata or {}).get("artefact_github_urls") or {}
        for artefact in mission.artefacts:
            name = Path(artefact).name
            gh_url = github_urls.get(artefact)
            if gh_url:
                # Permanent: hosted on GitHub, reachable even with the PC off.
                lines.append(f"- Capture (lien permanent) : {gh_url}")
            elif report_base_url:
                lines.append(
                    f"- Capture : {report_base_url}/v1/missions/{mission.mission_id}/artifacts/{name}"
                )
            else:
                lines.append(f"- Capture (locale, pas encore accessible par lien) : {artefact}")
        lines.append("")

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars/token) for BudgetGuard accounting."""
    return max(1, len(text or "") // 4)


__all__ = [
    "MissionEngine",
    "SystemAskAgent",
    "_default_steps",
    "coding_pr_steps",
    "research_steps",
    "improve_steps",
    "_build_report",
    "_estimate_tokens",
]
