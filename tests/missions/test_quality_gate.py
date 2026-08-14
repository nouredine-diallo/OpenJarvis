"""Tests for the adaptive quality gate (Brique 5).

The code gate runs real containers on purpose: its entire value is that
lint/security/tests are *actually executed* rather than asserted, and only
a real run can prove that. That approach immediately paid for itself --
two false positives that would have failed every well-formed project were
found this way (see the discriminates test).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, List

import pytest

from openjarvis.missions.quality_gate import (
    GATE_FULL,
    GATE_LIGHT,
    KIND_CODE,
    KIND_CONVERSATIONAL,
    KIND_RESEARCH,
    KIND_UI,
    GateCheck,
    GateResult,
    classify_mission,
    gate_level_for,
    run_code_checks,
    run_quality_gate,
    run_research_checks,
    run_ui_checks,
)


@dataclass
class FakeStep:
    result: str = ""
    required_capabilities: List[str] = field(default_factory=list)


@dataclass
class FakeMission:
    goal: str = ""
    steps: List[FakeStep] = field(default_factory=list)
    report: str = ""
    artefacts: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


requires_docker = pytest.mark.skipif(not _docker_ok(), reason="Docker unavailable")


class TestClassification:
    def test_coding_capability_means_code_gate(self):
        m = FakeMission(steps=[FakeStep(required_capabilities=["coding"])])
        assert classify_mission(m) == KIND_CODE

    def test_tests_capability_means_code_gate(self):
        m = FakeMission(steps=[FakeStep(required_capabilities=["tests"])])
        assert classify_mission(m) == KIND_CODE

    def test_preview_capability_means_ui_gate(self):
        m = FakeMission(steps=[FakeStep(required_capabilities=["preview"])])
        assert classify_mission(m) == KIND_UI

    def test_research_goal_means_research_gate(self):
        assert classify_mission(FakeMission(goal="Compare les solutions X et Y")) == KIND_RESEARCH

    def test_plain_question_is_conversational(self):
        assert classify_mission(FakeMission(goal="Bonjour, ça va ?")) == KIND_CONVERSATIONAL

    def test_capabilities_win_over_goal_wording(self):
        """Declared capabilities are structural truth; goal text is not."""
        m = FakeMission(goal="compare deux approches", steps=[FakeStep(required_capabilities=["coding"])])
        assert classify_mission(m) == KIND_CODE

    def test_gate_level_is_proportional(self):
        assert gate_level_for(KIND_CONVERSATIONAL) == GATE_LIGHT
        for kind in (KIND_CODE, KIND_RESEARCH, KIND_UI):
            assert gate_level_for(kind) == GATE_FULL


class TestResearchGate:
    def test_cited_synthesis_passes(self):
        m = FakeMission(
            steps=[FakeStep(result="Leon est self-hosted [1].")],
            report="## Sources\n[1] Leon — https://getleon.ai/",
        )
        assert all(c.passed for c in run_research_checks(m))

    def test_uncited_answer_fails(self):
        m = FakeMission(steps=[FakeStep(result="Il existe plein de solutions.")])
        checks = run_research_checks(m)
        assert any(not c.passed for c in checks)
        assert any("citation" in c.name for c in checks)

    def test_citations_without_source_list_fails(self):
        m = FakeMission(steps=[FakeStep(result="Affirmation [1].")], report="pas de liste")
        assert any(not c.passed and "sources" in c.name for c in run_research_checks(m))


class TestUiGate:
    def test_screenshot_artefact_passes(self):
        assert run_ui_checks(FakeMission(artefacts=["/tmp/shot.png"]))[0].passed is True

    def test_no_visual_artefact_fails(self):
        assert run_ui_checks(FakeMission(artefacts=["/tmp/log.txt"]))[0].passed is False

    def test_no_artefacts_at_all_fails(self):
        assert run_ui_checks(FakeMission())[0].passed is False


class TestGateResult:
    def test_passed_requires_every_check(self):
        r = GateResult(checks=[GateCheck("a", True), GateCheck("b", False)])
        assert r.passed is False

    def test_markdown_lists_each_check(self):
        r = GateResult(kind=KIND_CODE, checks=[GateCheck("lint", True, "OK")])
        assert "lint" in r.as_markdown()
        assert "✅" in r.as_markdown()

    def test_skipped_gate_is_reported_as_skipped(self):
        r = GateResult(skipped_reason="pas de projet")
        assert "ignoré" in r.as_markdown()


class TestEntryPoint:
    def test_conversational_mission_runs_no_heavy_checks(self):
        """Rigueur proportionnelle: a chat answer must not trigger a
        container run."""
        result = run_quality_gate(FakeMission(goal="salut"))
        assert result.kind == KIND_CONVERSATIONAL
        assert result.level == GATE_LIGHT
        assert result.checks == []

    def test_code_mission_without_project_dir_is_skipped_explicitly(self):
        m = FakeMission(steps=[FakeStep(required_capabilities=["coding"])])
        result = run_quality_gate(m, project_dir="")
        assert result.kind == KIND_CODE
        assert result.skipped_reason

    def test_sandbox_checks_can_be_disabled(self):
        m = FakeMission(steps=[FakeStep(required_capabilities=["coding"])])
        result = run_quality_gate(m, project_dir="/tmp", enable_sandbox_checks=False)
        assert result.skipped_reason

    def test_internal_error_surfaces_as_failed_check(self, monkeypatch):
        """A crashing gate must never read as a passing gate."""
        import openjarvis.missions.quality_gate as qg

        monkeypatch.setattr(
            qg, "run_research_checks", lambda m: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        result = run_quality_gate(FakeMission(goal="compare A et B"))
        assert result.passed is False


class TestCodeGateNoDocker:
    def test_missing_project_dir_fails(self):
        checks = run_code_checks("/does/not/exist")
        assert checks[0].passed is False

    def test_no_docker_fails_rather_than_skipping(self, monkeypatch, tmp_path):
        """"We could not verify" must never be reported as "verified"."""
        import openjarvis.missions.quality_gate as qg

        monkeypatch.setattr(qg, "_docker_available", lambda: False)
        checks = run_code_checks(str(tmp_path))
        assert checks[0].passed is False
        assert "Docker" in checks[0].detail


@requires_docker
class TestCodeGateReal:
    def _write_project(self, root, calc: str, test: str):
        root.mkdir(parents=True, exist_ok=True)
        (root / "calc.py").write_text(calc)
        (root / "test_calc.py").write_text(test)

    def test_discriminates_clean_from_broken_project(self, tmp_path):
        """The two false positives this caught on first run: ruff died
        trying to write its cache to the read-only mount, and bandit
        flagged the asserts *inside test files* -- either one would have
        failed every well-formed project."""
        good = tmp_path / "good"
        self._write_project(
            good,
            "def add(a, b):\n    return a + b\n",
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        assert all(c.passed for c in run_code_checks(str(good)))

        bad = tmp_path / "bad"
        self._write_project(
            bad,
            "import subprocess\ndef add(a,b):\n    subprocess.call('ls', shell=True)\n    return a - b\n",
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        assert all(not c.passed for c in run_code_checks(str(bad)))

    def test_project_without_tests_does_not_count_as_verified(self, tmp_path):
        """pytest exits 5 on "no tests collected" -- that is an absence of
        evidence, not evidence of correctness."""
        root = tmp_path / "notests"
        root.mkdir()
        (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        checks = run_code_checks(str(root))
        pytest_check = next(c for c in checks if "pytest" in c.name)
        assert pytest_check.passed is False
        assert "aucun test" in pytest_check.detail

    def test_pycache_is_not_copied_into_the_sandbox(self, tmp_path):
        """A stale __pycache__ can make tests pass against code that no
        longer exists (spec §4.3 point 4)."""
        root = tmp_path / "proj"
        self._write_project(
            root,
            "def add(a, b):\n    return a + b\n",
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        )
        stale = root / "__pycache__"
        stale.mkdir()
        (stale / "calc.cpython-312.pyc").write_bytes(b"stale garbage")

        checks = run_code_checks(str(root))
        assert all(c.passed for c in checks)
