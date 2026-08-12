"""Tests for the Mission Engine (Phase 4 / D10).

Covers the store, the verifier, and the engine's crash-recovery contract:
a mission interrupted mid-flight resumes from its last checkpoint on a new
engine instance (the "crash simulé" criterion).
"""

from __future__ import annotations

import time

import pytest

from openjarvis.core.events import EventBus
from openjarvis.missions import (
    MissionEngine,
    MissionStore,
    MissionStatus,
)
from openjarvis.missions.engine import _build_report, _default_steps, coding_pr_steps
from openjarvis.missions.types import Mission, MissionEvent, MissionStep
from openjarvis.missions.verifier import run_verification


@pytest.fixture
def store(tmp_path):
    s = MissionStore(tmp_path / "missions.db")
    yield s
    s.close()


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeSystem:
    """Deterministic stand-in for JarvisSystem.ask() that can fail per step."""

    def __init__(self, outputs=None, fail_substrings=()):
        self.outputs = list(outputs or [])
        self.fail_substrings = tuple(fail_substrings)
        self.calls = []

    def ask(self, query, *, context=True, **kwargs):
        self.calls.append(query)
        if any(f in query for f in self.fail_substrings):
            raise RuntimeError("boom")
        idx = len(self.calls) - 1
        content = self.outputs[min(idx, len(self.outputs) - 1)] if self.outputs else "done"
        return {"content": content}


class FakeCodingAgent:
    """Stand-in for the Phase 5 coding agent (OpenCodeAgent wrapper)."""

    def __init__(self, output="Patch appliqué, tests verts, PR #12 créée."):
        self.output = output
        self.calls = []

    def run(self, prompt):
        self.calls.append(prompt)
        return type("AgentResult", (), {"content": self.output})()


class FakeHeavyAgent:
    """Stand-in for ClaudeSubscriptionAgent (the frontier/heavy tier)."""

    def __init__(self, output="Review approfondie : rien à signaler.", fail=False, error=False):
        self.output = output
        self.fail = fail
        self.error = error
        self.calls = []

    def run(self, prompt):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("claude CLI unavailable")
        metadata = {"error": True} if self.error else {}
        return type("AgentResult", (), {"content": "" if self.error else self.output, "metadata": metadata})()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_checkpoint(store):
    m = Mission(
        mission_id="abc123",
        goal="Rechercher le président du Sénégal",
        created_at=time.time(),
        updated_at=time.time(),
        steps=_default_steps("Rechercher le président du Sénégal"),
    )
    store.create_mission(m)
    m.steps[0].status = "succeeded"
    m.steps[0].result = "Bassirou Diomaye Faye"
    m.status = MissionStatus.SUCCEEDED.value
    store.save_mission(m)

    loaded = store.get_mission("abc123")
    assert loaded is not None
    assert loaded.goal == m.goal
    assert loaded.status == MissionStatus.SUCCEEDED.value
    assert loaded.steps[0].result == "Bassirou Diomaye Faye"

    store.delete_mission("abc123")
    assert store.get_mission("abc123") is None


def test_store_list_and_inflight(store):
    for i in range(3):
        m = Mission(mission_id=f"m{i}", goal=f"g{i}", created_at=time.time(), updated_at=time.time())
        store.create_mission(m)
    m1 = store.get_mission("m1")
    m1.status = MissionStatus.SUCCEEDED.value
    store.save_mission(m1)

    assert {x.mission_id for x in store.list_missions()} == {"m0", "m1", "m2"}
    assert {x.mission_id for x in store.list_missions(status="succeeded")} == {"m1"}
    assert {x.mission_id for x in store.list_inflight()} == {"m0", "m2"}


def test_store_audit_events(store):
    store.create_mission(Mission(mission_id="m1", goal="g", created_at=1.0, updated_at=1.0))
    store.append_event(MissionEvent("m1", time.time(), "created", {}))
    store.append_event(MissionEvent("m1", time.time(), "succeeded", {"verified": True}))
    events = store.list_events("m1")
    assert [e.event_type for e in events] == ["created", "succeeded"]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verifier_rejects_empty_mission():
    m = Mission(mission_id="x", goal="", created_at=1.0, updated_at=1.0)
    v = run_verification(m)
    assert v["verified"] is False
    assert v["checks"]["has_goal"] is False


def test_verifier_accepts_complete_mission():
    m = Mission(
        mission_id="x",
        goal="Faire X",
        created_at=1.0,
        updated_at=1.0,
        status="succeeded",
        report="Objectif atteint avec preuves détaillées dans le rapport final.",
        steps=[
            MissionStep(
                index=0,
                title="A",
                status="succeeded",
                result="Résultat détaillé avec sources et preuves multiples.",
            )
        ],
    )
    v = run_verification(m)
    assert v["verified"] is True


def test_verifier_flags_missing_step_result():
    m = Mission(
        mission_id="x",
        goal="Faire X",
        created_at=1.0,
        updated_at=1.0,
        status="succeeded",
        report="Fait.",
        steps=[MissionStep(index=0, title="A", status="succeeded", result="")],
    )
    v = run_verification(m)
    assert v["checks"]["all_steps_have_results"] is False
    assert v["verified"] is False


# ---------------------------------------------------------------------------
# Engine — launch, execute, events, resume-after-crash
# ---------------------------------------------------------------------------


def test_launch_executes_and_finishes(store, event_bus):
    system = FakeSystem(["Premier résultat.", "Second résultat."])
    engine = MissionEngine(store, system, event_bus=event_bus, max_steps=2)
    engine.start()
    try:
        mission = engine.launch(
            "Mission de test",
            steps=[
                MissionStep(index=0, title="Étape 1", prompt="Fais 1"),
                MissionStep(index=1, title="Étape 2", prompt="Fais 2"),
            ],
            requested_by="telegram:1234",
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded", "succeeded"]
        assert done.report and "Mission de test" in done.report
        assert len(system.calls) == 2
        # MISSION_END is published after the terminal checkpoint is saved, so
        # wait for it rather than racing the worker.
        assert _wait_until(
            lambda: "mission_end" in [e.event_type for e in event_bus.history]
        )
        types = [e.event_type for e in event_bus.history]
        assert "mission_created" in types
        assert "mission_start" in types
        assert "mission_step_start" in types
        assert "mission_step_end" in types
        assert "mission_end" in types
        audit = [e.event_type for e in store.list_events(mission.mission_id)]
        assert "succeeded" in audit
    finally:
        engine.stop()


def test_failed_step_marks_mission_failed(store):
    system = FakeSystem(["ok"], fail_substrings=("Fais",))
    engine = MissionEngine(store, system, backoff_base=0.0)
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[MissionStep(index=0, title="S", prompt="Fais")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.FAILED.value
        assert "boom" in done.report
    finally:
        engine.stop()


def test_autonomy_zero_pauses_after_step(store):
    system = FakeSystem(["ok"])
    engine = MissionEngine(store, system)
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            autonomy_level=0,
            steps=[
                MissionStep(index=0, title="S1", prompt="Fais 1"),
                MissionStep(index=1, title="S2", prompt="Fais 2"),
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).status == "paused")
        done = engine.status(mission.mission_id)
        assert done.steps[0].status == "succeeded"
        assert done.steps[1].status == "pending"
        # resume -> runs step 2 then pauses again
        engine.resume(mission.mission_id)
        assert _wait_until(
            lambda: engine.status(mission.mission_id).status == "paused"
        )
        assert engine.status(mission.mission_id).steps[1].status == "succeeded"
    finally:
        engine.stop()


def test_budget_guard_pauses_mission(store):
    system = FakeSystem(["r" * 400])  # ~100 tokens per result
    engine = MissionEngine(store, system, max_budget_tokens=150, max_steps=5)
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[
                MissionStep(index=i, title=f"S{i}", prompt="Fais")
                for i in range(5)
            ],
        )
        assert _wait_until(
            lambda: engine.status(mission.mission_id).status == "paused"
        )
        done = engine.status(mission.mission_id)
        assert done.metadata.get("budget_exceeded") is True
    finally:
        engine.stop()


def test_crash_recovery_resumes_from_checkpoint(store):
    """The Phase 4 criterion: crash mid-mission, resume identically.

    We reconstruct the exact on-disk state a crash leaves behind (mission
    ``RUNNING``, step 0 checkpointed, step 1 mid-flight) and prove a fresh
    engine resumes from it — deterministic, no worker races.
    """
    mid_flight = Mission(
        mission_id="crash1",
        goal="Mission interrompue",
        created_at=time.time(),
        updated_at=time.time(),
        status=MissionStatus.RUNNING.value,
        max_steps=5,
        current_step=1,
        steps=[
            MissionStep(
                index=0,
                title="Étape 0",
                status="succeeded",
                result="Résultat de l'étape 0 déjà sauvegardé.",
                finished_at=time.time(),
            ),
            MissionStep(index=1, title="Étape 1", status="running"),
            MissionStep(index=2, title="Étape 2", status="pending"),
        ],
    )
    store.create_mission(mid_flight)
    store.append_event(
        MissionEvent("crash1", time.time(), "created", {"goal": mid_flight.goal})
    )

    # New engine over the same store = server restart.
    engine = MissionEngine(store, FakeSystem(["r1", "r2"]), max_steps=5)
    engine.start()
    try:
        assert _wait_until(lambda: engine.status("crash1").is_terminal)
        done = engine.status("crash1")
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded", "succeeded", "succeeded"]
        assert done.current_step == len(done.steps)
        audit = [e.event_type for e in store.list_events("crash1")]
        assert "restarted_after_crash" in audit
    finally:
        engine.stop()


def test_build_report_includes_steps():
    m = Mission(
        mission_id="abc",
        goal="G",
        created_at=1.0,
        updated_at=1.0,
        steps=[
            MissionStep(index=0, title="A", status="succeeded", result="res a"),
            MissionStep(index=1, title="B", status="pending"),
        ],
    )
    report = _build_report(m)
    assert "abc" in report
    assert "res a" in report
    assert "[OK]" in report


# ---------------------------------------------------------------------------
# Engine — worker capabilities gate (D12: WAITING_FOR_WORKER)
# ---------------------------------------------------------------------------


def _mission_with_capability(mission_id, capability, n_steps=2):
    return Mission(
        mission_id=mission_id,
        goal="Mission avec capacité requise",
        created_at=time.time(),
        updated_at=time.time(),
        max_steps=n_steps,
        steps=[
            MissionStep(index=0, title="Étape 0", prompt="Fais 0"),
            MissionStep(
                index=1,
                title="Étape 1",
                prompt="Fais 1",
                required_capabilities=[capability],
            ),
        ],
    )


def test_missing_capability_sets_waiting_for_worker(store, event_bus):
    system = FakeSystem(["r0", "r1"])
    engine = MissionEngine(store, system, event_bus=event_bus)
    engine.start()
    try:
        m = _mission_with_capability("wait1", "gpu")
        store.create_mission(m)
        engine.submit("wait1")
        assert _wait_until(
            lambda: engine.status("wait1").status
            == MissionStatus.WAITING_FOR_WORKER.value
        )
        # The audit event is appended just after the status checkpoint, so
        # wait for both instead of racing the worker.
        assert _wait_until(
            lambda: "waiting_for_worker"
            in [e.event_type for e in store.list_events("wait1")]
        )
        done = engine.status("wait1")
        assert done.steps[0].status == "succeeded"
        assert done.steps[1].status == "pending"
        assert done.metadata["waiting_for_worker"]["missing_capabilities"] == ["gpu"]
        assert "waiting_for_worker" in [e.event_type for e in store.list_events("wait1")]
    finally:
        engine.stop()


def test_capability_registered_resumes_mission(store):
    system = FakeSystem(["r0", "r1"])
    engine = MissionEngine(store, system)
    engine.start()
    try:
        m = _mission_with_capability("wait2", "gpu")
        store.create_mission(m)
        engine.submit("wait2")
        assert _wait_until(
            lambda: engine.status("wait2").status
            == MissionStatus.WAITING_FOR_WORKER.value
        )
        # A GPU worker comes online -> mission resumes and completes.
        resumed = engine.register_capabilities(["gpu", "docker"])
        assert "wait2" in resumed
        assert _wait_until(lambda: engine.status("wait2").is_terminal)
        done = engine.status("wait2")
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded", "succeeded"]
        audit = [e.event_type for e in store.list_events("wait2")]
        assert "worker_online" in audit
    finally:
        engine.stop()


def test_steps_without_capability_never_wait(store):
    system = FakeSystem(["r0", "r1"])
    engine = MissionEngine(store, system, worker_capabilities=["terminal"])
    engine.start()
    try:
        m = _mission_with_capability("nowait", "terminal")
        store.create_mission(m)
        engine.submit("nowait")
        assert _wait_until(lambda: engine.status("nowait").is_terminal)
        assert engine.status("nowait").status == MissionStatus.SUCCEEDED.value
    finally:
        engine.stop()


def test_config_accepts_worker_capabilities():
    from openjarvis.core.config import MissionsConfig

    cfg = MissionsConfig(enabled=True, worker_capabilities=["terminal", "docker"])
    assert cfg.worker_capabilities == ["terminal", "docker"]


def test_notify_includes_report_link(store):
    messages = []
    system = FakeSystem(["r0", "r1"])
    engine = MissionEngine(
        store,
        system,
        notifier=lambda target, title, msg: messages.append((target, title, msg)),
        report_base_url="https://jarvisland.duckdns.org",
    )
    engine.start()
    try:
        mission = engine.launch(
            "Mission notif",
            steps=[
                MissionStep(index=0, title="S1", prompt="Fais 1"),
                MissionStep(index=1, title="S2", prompt="Fais 2"),
            ],
            requested_by="telegram:6468865487",
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        assert _wait_until(lambda: bool(messages)), "expected a notification"
        target, title, body = messages[-1]
        assert target == "telegram:6468865487"
        assert title == "MISSION TERMINÉE"
        assert f"/v1/missions/{mission.mission_id}" in body
        assert "https://jarvisland.duckdns.org" in body
    finally:
        engine.stop()


def test_coding_step_dispatches_to_coding_agent(store):
    """Phase 5: a step with the ``coding`` capability runs through the coding
    agent (not a plain LLM answer), and non-coding steps still use the system."""
    system = FakeSystem(["Réponse LLM étape 1"])
    coder = FakeCodingAgent()
    engine = MissionEngine(store, system, coding_agent=coder, max_steps=3,
                           worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch(
            "Améliorer la page d'accueil",
            steps=[
                MissionStep(index=0, title="Analyser", prompt="Analyse le repo"),
                MissionStep(
                    index=1,
                    title="Implémenter",
                    prompt="Implémente la nouvelle page d'accueil puis pousse une PR",
                    required_capabilities=["coding"],
                ),
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded", "succeeded"]
        # The coding step was handled by the coding agent, not the system.
        assert len(coder.calls) == 1
        assert "page d'accueil" in coder.calls[0]
        assert len(system.calls) == 1
        assert "Patch appliqué" in done.steps[1].result
    finally:
        engine.stop()


def test_coding_pr_steps_plan_shape():
    steps = coding_pr_steps("Améliore la page d'accueil")
    assert [s.title for s in steps] == [
        "Setup",
        "Implement",
        "Test",
        "Review",
        "Ship",
    ]
    assert all(s.required_capabilities == ["coding"] for s in steps)
    assert [s.index for s in steps] == [0, 1, 2, 3, 4]
    assert all("Améliore la page d'accueil" in s.prompt for s in steps)
    # Only Review prefers the frontier tier -- the rest stay on the fast/free
    # coding agent by default.
    assert [s.prefer_heavy for s in steps] == [False, False, False, True, False]


def test_prefer_heavy_step_dispatches_to_heavy_agent(store):
    """A prefer_heavy step routes to the heavy agent, not the coding agent."""
    coder = FakeCodingAgent()
    heavy = FakeHeavyAgent()
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder, heavy_agent=heavy,
        worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Ajoute un endpoint /health",
            steps=[
                MissionStep(
                    index=0, title="Review", prompt="Relis le diff",
                    required_capabilities=["coding"], prefer_heavy=True,
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert len(heavy.calls) == 1
        assert len(coder.calls) == 0
        assert "rien à signaler" in done.steps[0].result
    finally:
        engine.stop()


def test_prefer_heavy_falls_back_when_agent_missing(store):
    """No heavy_agent configured -> the step still runs, via coding_agent."""
    coder = FakeCodingAgent()
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[
                MissionStep(
                    index=0, title="Review", prompt="Relis le diff",
                    required_capabilities=["coding"], prefer_heavy=True,
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert len(coder.calls) == 1
    finally:
        engine.stop()


def test_prefer_heavy_falls_back_on_error(store):
    """heavy_agent crashing or returning an error never fails the step --
    the soft preference degrades to coding_agent instead."""
    coder = FakeCodingAgent()
    for heavy in (FakeHeavyAgent(fail=True), FakeHeavyAgent(error=True)):
        engine = MissionEngine(
            store, FakeSystem(), coding_agent=coder, heavy_agent=heavy,
            worker_capabilities=["coding"],
        )
        engine.start()
        try:
            mission = engine.launch(
                "Mission",
                steps=[
                    MissionStep(
                        index=0, title="Review", prompt="Relis le diff",
                        required_capabilities=["coding"], prefer_heavy=True,
                    )
                ],
            )
            assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
            done = engine.status(mission.mission_id)
            assert done.status == MissionStatus.SUCCEEDED.value
            assert "Patch appliqué" in done.steps[0].result
        finally:
            engine.stop()


def test_coding_pr_mission_runs_5_checkpointed_steps(store):
    """kind='coding_pr' auto-plans 5 steps instead of one mega-step, and the
    engine checkpoints (saves) the mission after each one succeeds."""
    coder = FakeCodingAgent()
    engine = MissionEngine(store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch("Ajoute un endpoint /health", kind="coding_pr")
        assert len(mission.steps) == 5
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded"] * 5
        # Checkpointed: reloading straight from the store shows the same
        # per-step results (not just the in-memory object).
        reloaded = store.get_mission(mission.mission_id)
        assert [s.result for s in reloaded.steps] == [s.result for s in done.steps]
        assert len(coder.calls) == 5
    finally:
        engine.stop()


def test_coding_step_forwards_prior_step_context(store):
    """A later checkpointed coding step is told what earlier ones already
    did, so it does not redo git operations (redundant clone/branch/rebase)."""
    coder = FakeCodingAgent()
    engine = MissionEngine(store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch("Ajoute un endpoint /health", kind="coding_pr")
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        # Step 0 (Setup) has no prior context.
        assert "étapes déjà terminées" not in coder.calls[0]
        # Every later step references the earlier ones' results.
        for i in range(1, 5):
            assert "étapes déjà terminées" in coder.calls[i]
            assert f"Étape {i - 1}" in coder.calls[i]
            assert coder.output in coder.calls[i]
    finally:
        engine.stop()


def test_coding_step_falls_back_without_agent(store):
    system = FakeSystem(["Réponse LLM même pour l'étape coding"])
    engine = MissionEngine(store, system, max_steps=2, worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[
                MissionStep(
                    index=0,
                    title="Coder sans agent",
                    prompt="Implémente X",
                    required_capabilities=["coding"],
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert "Réponse LLM" in done.steps[0].result
    finally:
        engine.stop()
