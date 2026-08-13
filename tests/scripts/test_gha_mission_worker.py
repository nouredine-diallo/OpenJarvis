"""Tests for the GitHub Actions cloud-fallback mission worker script.

This runs unattended inside a CI job with no human watching -- the
failure mode that matters is a network/API error crashing the job
without ever reporting back to the control plane, which would leave a
mission silently stuck at PENDING forever.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gha_mission_worker.py"
_spec = importlib.util.spec_from_file_location("gha_mission_worker", _SCRIPT_PATH)
gha_mission_worker = importlib.util.module_from_spec(_spec)
sys.modules["gha_mission_worker"] = gha_mission_worker
_spec.loader.exec_module(gha_mission_worker)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ask_groq_strips_think_tags(monkeypatch):
    payload = json.dumps(
        {"choices": [{"message": {"content": "<think>reasoning...</think>Dakar"}}]}
    ).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )
    answer = gha_mission_worker.ask_groq("fake-key", "capital of Senegal?")
    assert answer == "Dakar"


def test_report_completion_sends_expected_payload(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data.decode())
        return _FakeResponse(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    gha_mission_worker.report_completion(
        "https://example.workers.dev", "shh", "mission-1", "SUCCEEDED", "Dakar"
    )
    assert captured["url"] == "https://example.workers.dev/missions/mission-1/complete"
    assert captured["headers"]["x-control-plane-secret"] == "shh"
    assert captured["body"] == {"status": "SUCCEEDED", "report": "Dakar"}


def test_main_reports_failure_when_groq_errors(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("groq down")

    reported = {}

    def _fake_report_completion(url, secret, mission_id, status, report):
        reported.update(mission_id=mission_id, status=status, report=report)

    monkeypatch.setenv("MISSION_ID", "m1")
    monkeypatch.setenv("MISSION_GOAL", "anything")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://example.workers.dev")
    monkeypatch.setenv("CONTROL_PLANE_SHARED_SECRET", "secret")
    monkeypatch.setattr("urllib.request.urlopen", _raise)
    monkeypatch.setattr(gha_mission_worker, "report_completion", _fake_report_completion)

    exit_code = gha_mission_worker.main()

    assert exit_code == 1
    assert reported["mission_id"] == "m1"
    assert reported["status"] == "FAILED"
    assert "groq down" in reported["report"] or "URLError" in reported["report"]


def test_main_succeeds_and_reports_on_happy_path(monkeypatch):
    payload = json.dumps({"choices": [{"message": {"content": "Dakar"}}]}).encode()
    reported = {}

    monkeypatch.setenv("MISSION_ID", "m2")
    monkeypatch.setenv("MISSION_GOAL", "capital of Senegal?")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://example.workers.dev")
    monkeypatch.setenv("CONTROL_PLANE_SHARED_SECRET", "secret")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )
    monkeypatch.setattr(
        gha_mission_worker,
        "report_completion",
        lambda url, secret, mission_id, status, report: reported.update(
            mission_id=mission_id, status=status, report=report
        ),
    )

    exit_code = gha_mission_worker.main()

    assert exit_code == 0
    assert reported == {"mission_id": "m2", "status": "SUCCEEDED", "report": "Dakar"}
