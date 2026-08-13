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
from openjarvis.missions.engine import (
    SystemAskAgent,
    _build_report,
    _default_steps,
    coding_pr_steps,
    improve_steps,
    research_steps,
)
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


def test_verifier_flags_step_that_honestly_reports_being_blocked():
    """Reproduces the exact live failure (2026-08-12): a coding_pr_steps
    mission where every step honestly says it couldn't do its job
    (permissions/sandbox refused writes) -- each step still has
    non-empty result text, so it gets checkpointed 'succeeded', but the
    mission must NOT be allowed to report overall success."""
    m = Mission(
        mission_id="x",
        goal="Ajoute des tests",
        created_at=1.0,
        updated_at=1.0,
        status="succeeded",
        report="Mission terminée.",
        steps=[
            MissionStep(index=0, title="Setup", status="succeeded", result="Les commandes git sont bloquées par les permissions."),
            MissionStep(index=1, title="Implement", status="succeeded", result="L'écriture du fichier a été refusée (permissions)."),
            MissionStep(index=2, title="Ship", status="succeeded", result="Rien n'a été réellement créé ni committé."),
        ],
    )
    v = run_verification(m)
    assert v["checks"]["no_blocked_steps"] is False
    assert v["verified"] is False


def test_mission_engine_fails_instead_of_succeeding_when_steps_are_blocked(store):
    """End-to-end version of the check above: MissionEngine._finish() must
    gate the final status on verification, not just checkpoint whatever
    non-empty text each step produced (this used to report MISSION
    TERMINÉE even though nothing was actually done)."""
    system = FakeSystem([
        "Les commandes git sont bloquées par les permissions sur ce chemin.",
        "Rien n'a été réellement créé : l'écriture a été refusée.",
    ])
    messages = []
    engine = MissionEngine(
        store, system,
        notifier=lambda target, title, msg: messages.append((target, title, msg)),
    )
    engine.start()
    try:
        mission = engine.launch(
            "Ajoute des tests",
            steps=[
                MissionStep(index=0, title="Setup", prompt="Fais 1"),
                MissionStep(index=1, title="Ship", prompt="Fais 2"),
            ],
            requested_by="telegram:1",
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.FAILED.value
        assert done.verification["verified"] is False
        assert done.verification["checks"]["no_blocked_steps"] is False
        # Every step still shows succeeded (the LLM did respond) -- it's
        # the mission-level status that must reflect the real outcome.
        assert [s.status for s in done.steps] == ["succeeded", "succeeded"]
        assert _wait_until(lambda: bool(messages))
        assert messages[-1][1] == "MISSION NON VÉRIFIÉE"
    finally:
        engine.stop()


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


def test_build_report_includes_confidence_and_preview_url():
    """Spec §31/§33: the report must carry a confidence level and, when a
    screenshot exists, a fetchable preview URL -- not just a step log."""
    m = Mission(
        mission_id="abc",
        goal="G",
        created_at=1.0,
        updated_at=1.0,
        status="succeeded",
        verification={"verified": True, "checks": {"has_goal": True, "has_steps": True}},
        artefacts=["/home/land/.openjarvis/mission_artifacts/abc/final.png"],
        steps=[MissionStep(index=0, title="A", status="succeeded", result="res a")],
    )
    report = _build_report(m, report_base_url="https://jarvisland.duckdns.org")
    assert "confiance" in report
    assert "Élevée" in report
    assert "https://jarvisland.duckdns.org/v1/missions/abc/artifacts/final.png" in report


def test_build_report_confidence_low_when_a_check_failed():
    m = Mission(
        mission_id="abc",
        goal="G",
        created_at=1.0,
        updated_at=1.0,
        verification={"verified": False, "checks": {"has_goal": True, "no_blocked_steps": False}},
    )
    report = _build_report(m)
    assert "Faible" in report
    assert "no_blocked_steps" in report


# ---------------------------------------------------------------------------
# Engine — feedback / revision round (spec §32)
# ---------------------------------------------------------------------------


def test_give_feedback_appends_revision_round_and_resumes(store):
    coder = FakeCodingAgent()
    engine = MissionEngine(store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch(
            "Ajoute une page d'accueil",
            steps=coding_pr_steps("Ajoute une page d'accueil"),
            requested_by="telegram:7",
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        first_round = engine.status(mission.mission_id)
        assert first_round.status == MissionStatus.SUCCEEDED.value
        assert len(first_round.steps) == 5

        revised = engine.give_feedback(mission.mission_id, "Le bouton est trop gros")
        assert revised is not None
        assert len(revised.steps) == 10  # original 5 + a fresh 5-phase round
        assert revised.status == MissionStatus.PENDING.value

        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded"] * 10
        assert done.metadata.get("feedback_rounds") == ["Le bouton est trop gros"]
        assert "trop gros" in done.report
        events = [e.event_type for e in store.list_events(mission.mission_id)]
        assert "feedback_received" in events
    finally:
        engine.stop()


def test_give_feedback_without_mission_id_targets_most_recent_succeeded(store):
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=FakeCodingAgent(), worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Ajoute une page",
            steps=coding_pr_steps("Ajoute une page"),
            requested_by="telegram:9",
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        revised = engine.give_feedback(None, "trop lent", requested_by="telegram:9")
        assert revised is not None
        assert revised.mission_id == mission.mission_id
    finally:
        engine.stop()


def test_give_feedback_ignored_for_non_coding_mission(store):
    """Feedback only makes sense on a mission that touched code -- a
    research mission's answer isn't something a revision round applies to."""
    engine = MissionEngine(store, FakeSystem(["Réponse de recherche."]))
    engine.start()
    try:
        mission = engine.launch(
            "Quelle est la capitale du Sénégal ?",
            steps=[MissionStep(index=0, title="Réponse", prompt="Fais 1")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        unchanged = engine.give_feedback(mission.mission_id, "pas convaincant")
        assert unchanged is not None
        assert len(unchanged.steps) == 1  # nothing appended
        assert unchanged.status == MissionStatus.SUCCEEDED.value  # unchanged
    finally:
        engine.stop()


def test_give_feedback_returns_none_when_no_mission_found(store):
    engine = MissionEngine(store, FakeSystem())
    engine.start()
    try:
        assert engine.give_feedback(None, "quoi que ce soit", requested_by="telegram:404") is None
    finally:
        engine.stop()


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
    system = FakeSystem(["Résultat détaillé de l'étape 0.", "Résultat détaillé de l'étape 1."])
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
    system = FakeSystem(["Résultat détaillé de l'étape 0.", "Résultat détaillé de l'étape 1."])
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
    system = FakeSystem(["Résultat détaillé de l'étape 0.", "Résultat détaillé de l'étape 1."])
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
    system = FakeSystem(["Résultat détaillé de l'étape 0.", "Résultat détaillé de l'étape 1."])
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


def test_visual_proof_captured_and_sent_when_server_detected(store, monkeypatch):
    """A coding mission that leaves a dev server running gets a screenshot
    captured and sent as a photo after it succeeds."""
    calls = {"sent": []}

    def _fake_find(evidence):
        return "http://127.0.0.1:3000" if "listening" in evidence else None

    def _fake_capture(url, out_path, **kwargs):
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nfake")
        return out_path, "ok (0.1s)"

    monkeypatch.setattr("openjarvis.tools.screenshot.find_dev_server_url", _fake_find)
    monkeypatch.setattr("openjarvis.tools.screenshot.capture_screenshot", _fake_capture)

    def _photo_sender(target, path, caption):
        calls["sent"].append((target, path, caption))
        return True

    system = FakeSystem(["Serveur démarré, listening on port 3000."])
    engine = MissionEngine(
        store, system, photo_sender=_photo_sender, worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Crée une page d'accueil",
            steps=[
                MissionStep(
                    index=0, title="Ship", prompt="Fais 1",
                    required_capabilities=["coding"],
                )
            ],
            requested_by="telegram:1",
        )
        # _try_visual_proof now runs BEFORE the terminal-status checkpoint
        # save (see _finish's single-save fix) specifically so that once
        # is_terminal is true, every mutation -- including this photo send
        # and the artefact append -- has already happened. No separate
        # wait needed for calls["sent"].
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        assert len(calls["sent"]) == 1
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        target, path, caption = calls["sent"][0]
        assert target == "telegram:1"
        assert path.endswith("final.png")
        assert path in done.artefacts
        events = [e.event_type for e in store.list_events(mission.mission_id)]
        assert "visual_proof_captured" in events
    finally:
        engine.stop()


def test_artifact_backed_up_to_github_gives_permanent_report_link(store, monkeypatch):
    """When the GitHub backup push succeeds, the report's Preuves section
    links to the permanent github.com URL instead of the local tunnel/API
    URL -- so the link keeps working even with the PC off."""

    def _fake_find(evidence):
        return "http://127.0.0.1:3000" if "listening" in evidence else None

    def _fake_capture(url, out_path, **kwargs):
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nfake")
        return out_path, "ok (0.1s)"

    monkeypatch.setattr("openjarvis.tools.screenshot.find_dev_server_url", _fake_find)
    monkeypatch.setattr("openjarvis.tools.screenshot.capture_screenshot", _fake_capture)
    monkeypatch.setattr(
        "openjarvis.tools.artifact_backup.push_artifact",
        lambda local_path, mission_id: (
            f"https://github.com/nouredine-diallo/jarvis-artifacts/blob/main/"
            f"missions/{mission_id}/final.png"
        ),
    )

    system = FakeSystem(["Serveur démarré, listening on port 3000."])
    engine = MissionEngine(
        store, system,
        photo_sender=lambda t, p, c: True,
        worker_capabilities=["coding"],
        report_base_url="https://jarvisland.duckdns.org",
    )
    engine.start()
    try:
        mission = engine.launch(
            "Crée une page d'accueil",
            steps=[
                MissionStep(
                    index=0, title="Ship", prompt="Fais 1",
                    required_capabilities=["coding"],
                )
            ],
        )
        # The GitHub push runs in a background thread (must never block the
        # mission's own completion), so the permanent link lands in the
        # report asynchronously -- poll for it rather than a one-shot check.
        assert _wait_until(
            lambda: "jarvis-artifacts/blob/main" in (engine.status(mission.mission_id).report or "")
        )
        done = engine.status(mission.mission_id)
        assert "lien permanent" in done.report
        assert "jarvisland.duckdns.org" not in done.report
    finally:
        engine.stop()


def test_artifact_backup_failure_falls_back_to_local_report_url(store, monkeypatch):
    """If the GitHub push fails (offline, quota, etc.), the report still
    gets a usable link -- the local tunnel/API URL -- rather than nothing."""

    def _fake_find(evidence):
        return "http://127.0.0.1:3000" if "listening" in evidence else None

    def _fake_capture(url, out_path, **kwargs):
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\nfake")
        return out_path, "ok (0.1s)"

    monkeypatch.setattr("openjarvis.tools.screenshot.find_dev_server_url", _fake_find)
    monkeypatch.setattr("openjarvis.tools.screenshot.capture_screenshot", _fake_capture)
    monkeypatch.setattr(
        "openjarvis.tools.artifact_backup.push_artifact",
        lambda local_path, mission_id: None,
    )

    system = FakeSystem(["Serveur démarré, listening on port 3000."])
    engine = MissionEngine(
        store, system,
        photo_sender=lambda t, p, c: True,
        worker_capabilities=["coding"],
        report_base_url="https://jarvisland.duckdns.org",
    )
    engine.start()
    try:
        mission = engine.launch(
            "Crée une page d'accueil",
            steps=[
                MissionStep(
                    index=0, title="Ship", prompt="Fais 1",
                    required_capabilities=["coding"],
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert "jarvisland.duckdns.org" in done.report
        assert "lien permanent" not in done.report
    finally:
        engine.stop()


def test_visual_proof_skipped_when_no_server_detected(store, monkeypatch):
    monkeypatch.setattr(
        "openjarvis.tools.screenshot.find_dev_server_url", lambda evidence: None
    )
    calls = []
    system = FakeSystem(["Résultat détaillé sans aucun serveur web."])
    engine = MissionEngine(
        store, system,
        photo_sender=lambda t, p, c: calls.append((t, p, c)) or True,
        worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Ajoute une fonction utilitaire",
            steps=[
                MissionStep(
                    index=0, title="Ship", prompt="Fais 1",
                    required_capabilities=["coding"],
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        assert _wait_until(
            lambda: "visual_proof_skipped"
            in [e.event_type for e in store.list_events(mission.mission_id)]
        )
        assert calls == []
    finally:
        engine.stop()


def test_visual_proof_skipped_for_non_coding_mission(store, monkeypatch):
    """Research missions etc. never trigger a screenshot attempt at all --
    not even a dev-server probe."""
    probed = []
    monkeypatch.setattr(
        "openjarvis.tools.screenshot.find_dev_server_url",
        lambda evidence: probed.append(evidence) or None,
    )
    calls = []
    system = FakeSystem(["Réponse de recherche, aucun rapport avec du code."])
    engine = MissionEngine(
        store, system, photo_sender=lambda t, p, c: calls.append((t, p, c)) or True,
    )
    engine.start()
    try:
        mission = engine.launch(
            "Quelle est la capitale du Sénégal ?",
            steps=[MissionStep(index=0, title="Réponse", prompt="Fais 1")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        assert calls == []
        assert probed == []  # never even attempted detection
    finally:
        engine.stop()


def test_visual_proof_disabled_via_config(store, monkeypatch):
    probed = []
    monkeypatch.setattr(
        "openjarvis.tools.screenshot.find_dev_server_url",
        lambda evidence: probed.append(evidence) or "http://127.0.0.1:3000",
    )
    calls = []
    system = FakeSystem(["Résultat."])
    engine = MissionEngine(
        store, system,
        photo_sender=lambda t, p, c: calls.append((t, p, c)) or True,
        enable_visual_proof=False,
        worker_capabilities=["coding"],
    )
    engine.start()
    try:
        mission = engine.launch(
            "Crée une page",
            steps=[
                MissionStep(
                    index=0, title="Ship", prompt="Fais 1",
                    required_capabilities=["coding"],
                )
            ],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        assert calls == []
        assert probed == []
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


def test_prefer_heavy_tries_multiple_agents_in_order(store):
    """Multiple frontier tiers (e.g. Claude subscription + Gemini
    subscription) are tried in order -- the first one down/erroring
    doesn't stop the step, it just moves to the next candidate. This is
    the 'combine several LLMs, distribute the work' fallback chain."""
    coder = FakeCodingAgent()
    first_down = FakeHeavyAgent(fail=True)
    second_up = FakeHeavyAgent(output="Avis du deuxième modèle : diff propre.")
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder,
        heavy_agent=first_down, heavy_agents=[second_up],
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
        assert len(first_down.calls) == 1
        assert len(second_up.calls) == 1
        assert len(coder.calls) == 0
        assert "deuxième modèle" in done.steps[0].result
    finally:
        engine.stop()


def test_default_provider_failure_falls_back_to_heavy_agent(store):
    """A step with NO prefer_heavy still recovers if the default provider
    (system.ask -- e.g. Groq out of daily quota) fails: the frontier tier
    is tried as a last resort rather than failing the step outright."""
    system = FakeSystem(fail_substrings=("Fais",))
    heavy = FakeHeavyAgent(output="Réponse du tier de secours.")
    engine = MissionEngine(store, system, heavy_agent=heavy)
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[MissionStep(index=0, title="S", prompt="Fais quelque chose")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert len(heavy.calls) >= 1
        assert "tier de secours" in done.steps[0].result
    finally:
        engine.stop()


def test_default_provider_failure_tries_free_fallback_before_heavy(store):
    """default_fallback_agents (e.g. a sibling Groq model, its own
    separate free daily quota) is tried BEFORE ever spending
    Claude/Gemini subscription budget -- only if that also fails does the
    heavy tier get touched."""
    system = FakeSystem(fail_substrings=("Fais",))
    free_sibling = FakeHeavyAgent(output="Réponse d'un autre modèle Groq gratuit.")
    heavy = FakeHeavyAgent(output="Réponse Claude -- ne devrait jamais être appelée ici.")
    engine = MissionEngine(
        store, system, heavy_agent=heavy, default_fallback_agents=[free_sibling]
    )
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[MissionStep(index=0, title="S", prompt="Fais quelque chose")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert len(free_sibling.calls) == 1
        assert len(heavy.calls) == 0  # never touched -- free tier answered first
        assert "autre modèle Groq gratuit" in done.steps[0].result
    finally:
        engine.stop()


def test_default_fallback_agents_skipped_when_all_fail_then_heavy_used(store):
    """If every free_fallback candidate also fails, the heavy tier still
    gets its chance -- the free tier is an extra rung, not a replacement."""
    system = FakeSystem(fail_substrings=("Fais",))
    free_sibling = FakeHeavyAgent(fail=True)
    heavy = FakeHeavyAgent(output="Réponse Claude -- filet de secours ultime.")
    engine = MissionEngine(
        store, system, heavy_agent=heavy, default_fallback_agents=[free_sibling]
    )
    engine.start()
    try:
        mission = engine.launch(
            "Mission",
            steps=[MissionStep(index=0, title="S", prompt="Fais quelque chose")],
        )
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert len(free_sibling.calls) == 1
        assert len(heavy.calls) == 1
        assert "filet de secours ultime" in done.steps[0].result
    finally:
        engine.stop()


def test_prefer_heavy_step_skips_free_fallback_goes_straight_to_heavy(store):
    """A prefer_heavy step (e.g. code Review) wants quality, not just
    availability -- it must go straight to the heavy tier and must NOT
    try the free/fast sibling models first."""
    coder = FakeCodingAgent()
    free_sibling = FakeHeavyAgent(output="Réponse rapide mais pas assez rigoureuse.")
    heavy = FakeHeavyAgent(output="Review approfondie du tier frontier.")
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder, heavy_agent=heavy,
        default_fallback_agents=[free_sibling], worker_capabilities=["coding"],
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
        assert len(free_sibling.calls) == 0  # never tried -- prefer_heavy skips it
        assert len(heavy.calls) == 1
        assert "Review approfondie" in done.steps[0].result
    finally:
        engine.stop()


def test_system_ask_agent_adapts_run_interface(tmp_path):
    """SystemAskAgent adapts JarvisSystem.ask() to the .run(prompt) ->
    result-with-.content interface the fallback machinery expects."""

    class _FakeJarvisSystem:
        model = "groq/llama-3.1-8b-instant"

        def ask(self, query, *, context=True, **kwargs):
            return {"content": f"réponse à: {query}"}

    agent = SystemAskAgent(_FakeJarvisSystem())
    result = agent.run("bonjour")
    assert result.content == "réponse à: bonjour"
    assert not result.metadata.get("error")
    assert agent.label == "groq/llama-3.1-8b-instant"


def test_system_ask_agent_reports_error_on_exception():
    class _BrokenSystem:
        def ask(self, query, *, context=True, **kwargs):
            raise RuntimeError("quota épuisé")

    agent = SystemAskAgent(_BrokenSystem(), label="groq-sibling")
    result = agent.run("bonjour")
    assert result.metadata.get("error") is True
    assert result.content == ""


def test_research_steps_plan_shape():
    steps = research_steps("Quelle est la meilleure architecture pour X ?")
    assert [s.title for s in steps] == ["Recherche", "Vérification croisée", "Synthèse"]
    assert [s.index for s in steps] == [0, 1, 2]
    # Only the final synthesis prefers the frontier tier.
    assert [s.prefer_heavy for s in steps] == [False, False, True]
    assert all(s.required_capabilities == [] for s in steps)
    assert all("meilleure architecture" in s.prompt for s in steps)


def test_research_mission_runs_3_checkpointed_steps(store):
    system = FakeSystem(["Sources trouvées.", "Points confirmés/contradictoires.", "Réponse finale."])
    engine = MissionEngine(store, system)
    engine.start()
    try:
        mission = engine.launch("Quel est le meilleur ORM Python ?", kind="research")
        assert len(mission.steps) == 3
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded"] * 3
        assert len(system.calls) == 3
    finally:
        engine.stop()


def test_improve_steps_plan_shape():
    steps = improve_steps("Regarde mon app et améliore-la")
    assert [s.title for s in steps] == ["Analyse", "Proposition"]
    assert [s.pause_for_choice for s in steps] == [False, True]
    assert all(s.required_capabilities == ["coding"] for s in steps)


def test_improve_mission_pauses_for_choice_instead_of_executing(store):
    """The Proposition step succeeding does NOT continue the mission --
    it stops at WAITING_FOR_CHOICE. In particular none of the 5
    coding_pr_steps phases (Setup/Implement/Test/Review/Ship) exist yet --
    nothing gets built before the user picks an option (spec §27)."""
    coder = FakeCodingAgent()
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"]
    )
    engine.start()
    try:
        mission = engine.launch("Regarde mon app et améliore-la", kind="improve")
        assert _wait_until(
            lambda: engine.status(mission.mission_id).status == "waiting_for_choice"
        )
        # The audit event is appended just after the status checkpoint, so
        # wait for both instead of racing the worker (same pattern as the
        # WAITING_FOR_WORKER tests above).
        assert _wait_until(
            lambda: "waiting_for_choice"
            in [e.event_type for e in store.list_events(mission.mission_id)]
        )
        done = engine.status(mission.mission_id)
        assert [s.status for s in done.steps] == ["succeeded", "succeeded"]
        assert [s.title for s in done.steps] == ["Analyse", "Proposition"]
        assert len(done.steps) == 2  # nothing appended/executed yet
        assert len(coder.calls) == 2  # only Analyse + Proposition ran
    finally:
        engine.stop()


def test_choose_appends_coding_pr_steps_and_resumes(store):
    """MissionEngine.choose() appends the 5 coding_pr_steps for the chosen
    option and resumes -- the mission keeps its Analyse/Proposition history
    (their titles/count stay put; only new steps get appended after)."""
    coder = FakeCodingAgent()
    engine = MissionEngine(
        store, FakeSystem(), coding_agent=coder, worker_capabilities=["coding"]
    )
    engine.start()
    try:
        mission = engine.launch("Regarde mon app et améliore-la", kind="improve")
        assert _wait_until(
            lambda: engine.status(mission.mission_id).status == "waiting_for_choice"
        )
        pre_choice_results = [s.result for s in engine.status(mission.mission_id).steps]

        chosen = engine.choose(mission.mission_id, "Option 1 (les tests)")
        assert chosen is not None
        assert len(chosen.steps) == 7  # 2 (Analyse/Proposition) + 5 (coding_pr)
        assert [s.title for s in chosen.steps[:2]] == ["Analyse", "Proposition"]
        assert [s.title for s in chosen.steps[2:]] == [
            "Setup", "Implement", "Test", "Review", "Ship",
        ]
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
        done = engine.status(mission.mission_id)
        assert done.status == MissionStatus.SUCCEEDED.value
        assert [s.status for s in done.steps] == ["succeeded"] * 7
        # The original 2 steps' results are untouched by the later phases.
        assert [s.result for s in done.steps[:2]] == pre_choice_results
        assert done.metadata.get("chosen_option") == "Option 1 (les tests)"
        assert len(coder.calls) == 7  # Analyse+Proposition, then the 5 appended phases
        assert "Option choisie par l'utilisateur : Option 1" in coder.calls[2]
    finally:
        engine.stop()


def test_choose_without_mission_id_targets_most_recent_for_requester(store):
    system = FakeSystem(["Analyse.", "1. A\n2. B"])
    coder = FakeCodingAgent()
    engine = MissionEngine(store, system, coding_agent=coder, worker_capabilities=["coding"])
    engine.start()
    try:
        mission = engine.launch(
            "Regarde mon app", kind="improve", requested_by="telegram:42"
        )
        assert _wait_until(
            lambda: engine.status(mission.mission_id).status == "waiting_for_choice"
        )
        chosen = engine.choose(None, "la 1", requested_by="telegram:42")
        assert chosen is not None
        assert chosen.mission_id == mission.mission_id
        assert _wait_until(lambda: engine.status(mission.mission_id).is_terminal)
    finally:
        engine.stop()


def test_choose_returns_none_when_nothing_pending(store):
    engine = MissionEngine(store, FakeSystem())
    engine.start()
    try:
        assert engine.choose(None, "n'importe quoi", requested_by="telegram:999") is None
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
