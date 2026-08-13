"""Tests for the control-plane heartbeat sender (Phase A, PC-independence).

The failure mode that matters here isn't "the heartbeat succeeds" (that's
just an HTTP call) -- it's "a dead/unreachable control plane must never
raise into the caller", since this runs unattended in a background thread
inside the same process as the mission engine and the Telegram bot.
"""

from __future__ import annotations

import json
import time
import urllib.error

import pytest

from openjarvis.tools.control_plane_heartbeat import (
    ControlPlaneHeartbeat,
    send_heartbeat,
)


class _FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_heartbeat_success(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data.decode())
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    ok = send_heartbeat(
        "https://example.workers.dev", "shh-secret", ["coding", "tests"], worker_id="pc"
    )

    assert ok is True
    assert captured["url"] == "https://example.workers.dev/heartbeat"
    assert captured["method"] == "POST"
    assert captured["headers"]["x-control-plane-secret"] == "shh-secret"
    assert captured["body"] == {"worker_id": "pc", "capabilities": ["coding", "tests"]}


def test_send_heartbeat_strips_trailing_slash(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    send_heartbeat("https://example.workers.dev/", "secret", [])
    assert captured["url"] == "https://example.workers.dev/heartbeat"


def test_send_heartbeat_returns_false_on_non_2xx(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(500)
    )
    assert send_heartbeat("https://example.workers.dev", "secret", []) is False


def test_send_heartbeat_swallows_network_errors(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    assert send_heartbeat("https://example.workers.dev", "secret", []) is False


def test_background_thread_sends_multiple_heartbeats(monkeypatch):
    calls = []

    def _fake_urlopen(request, timeout=None):
        calls.append(time.time())
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    hb = ControlPlaneHeartbeat(
        "https://example.workers.dev", "secret", ["coding"], interval_seconds=0.05
    )
    hb.start()
    try:
        deadline = time.time() + 2.0
        while len(calls) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert len(calls) >= 3
    finally:
        hb.stop()


def test_disabled_when_unconfigured_does_not_start_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: calls.append(1) or _FakeResponse(200),
    )
    hb = ControlPlaneHeartbeat("", "", [])
    hb.start()
    time.sleep(0.1)
    hb.stop()
    assert calls == []


def test_stop_is_idempotent_and_bounded():
    hb = ControlPlaneHeartbeat("https://example.workers.dev", "secret", [], interval_seconds=10)
    hb.start()
    hb.stop()
    hb.stop()  # must not raise or hang
