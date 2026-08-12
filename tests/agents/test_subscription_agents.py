"""Tests for the headless-CLI subscription agents (frontier tier, D13/D14).

Found live and worth guarding against regressing: (1) claude CLI can exit
non-zero with the real error living in stdout JSON, not stderr -- the
naive version reported "Unknown error"; (2) gemini CLI is a Node shebang
script, so its subprocess needs the resolved binary's own directory on
PATH or execve fails at the shebang's `env node` lookup, independent of
whether the binary itself was *found* correctly.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from openjarvis.agents.claude_subscription import ClaudeSubscriptionAgent
from openjarvis.agents.gemini_subscription import GeminiSubscriptionAgent


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# ClaudeSubscriptionAgent
# ---------------------------------------------------------------------------


def test_claude_run_success_parses_result_and_cost(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )
    payload = json.dumps({"result": "Diff propre.", "total_cost_usd": 0.12, "num_turns": 2})

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout=payload)

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(workspace=str(tmp_path))
    result = agent.run("Relis le diff")
    assert result.content == "Diff propre."
    assert result.metadata["subscription_cost_estimate_usd"] == 0.12
    assert not result.metadata.get("error")


def test_claude_run_nonzero_exit_reads_error_from_stdout_json(monkeypatch, tmp_path):
    """Regression test: found live that claude -p prints its error object to
    stdout (not stderr) on a non-zero exit -- the earlier version reported
    "Unknown error" and threw away the real diagnostic."""
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )
    payload = json.dumps({"error": {"type": "Error", "message": "usage limit reached"}})

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout=payload, stderr="")

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(workspace=str(tmp_path))
    result = agent.run("Relis le diff")
    assert result.metadata["error"] is True
    assert "usage limit reached" in result.content
    assert "Unknown error" not in result.content


def test_claude_run_nonzero_exit_falls_back_to_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="permission denied")

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(workspace=str(tmp_path))
    result = agent.run("Relis le diff")
    assert result.metadata["error"] is True
    assert "permission denied" in result.content


def test_claude_run_nonzero_exit_no_output_at_all(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(workspace=str(tmp_path))
    result = agent.run("Relis le diff")
    assert result.metadata["error"] is True
    assert "exit code 1" in result.content


def test_claude_run_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(workspace=str(tmp_path), timeout=5)
    result = agent.run("Relis le diff")
    assert result.metadata["error_type"] == "timeout"


def test_claude_daily_budget_guard_blocks_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.claude_subscription.is_claude_subscription_available",
        lambda: True,
    )
    called = []

    def fake_run(cmd, **kwargs):
        called.append(cmd)
        return _FakeCompletedProcess(returncode=0, stdout=json.dumps({"result": "ok"}))

    monkeypatch.setattr("openjarvis.agents.claude_subscription.subprocess.run", fake_run)
    agent = ClaudeSubscriptionAgent(
        workspace=str(tmp_path),
        daily_budget_usd=0.05,
        usage_log_path=str(tmp_path / "usage.jsonl"),
    )
    monkeypatch.setattr(agent, "_today_spend", lambda: 1.0)
    result = agent.run("Relis le diff")
    assert result.metadata.get("budget_exceeded") is True
    assert not called  # never even spawned the subprocess


# ---------------------------------------------------------------------------
# GeminiSubscriptionAgent
# ---------------------------------------------------------------------------


def test_gemini_run_success_parses_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.gemini_subscription.is_gemini_subscription_available",
        lambda: True,
    )
    payload = json.dumps({"response": "Réponse du modèle.", "stats": {"tokens": 42}})

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0, stdout=payload)

    monkeypatch.setattr("openjarvis.agents.gemini_subscription.subprocess.run", fake_run)
    agent = GeminiSubscriptionAgent(workspace=str(tmp_path), gemini_bin="/fake/bin/gemini")
    result = agent.run("Question")
    assert result.content == "Réponse du modèle."
    assert not result.metadata.get("error")


def test_gemini_run_prepends_binary_dir_to_path(monkeypatch, tmp_path):
    """Regression test: found live that gemini is a `#!/usr/bin/env node`
    shebang script -- a systemd service's minimal PATH resolves the binary
    path correctly (find_gemini_bin) but execve still fails at `env node`
    unless that directory is actually on PATH. Reproduced and fixed live
    with this exact prepend."""
    monkeypatch.setattr(
        "openjarvis.agents.gemini_subscription.is_gemini_subscription_available",
        lambda: True,
    )
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeCompletedProcess(returncode=0, stdout=json.dumps({"response": "ok"}))

    monkeypatch.setattr("openjarvis.agents.gemini_subscription.subprocess.run", fake_run)
    fake_bin = tmp_path / "node_bin" / "gemini"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.write_text("#!/usr/bin/env node\n")
    agent = GeminiSubscriptionAgent(workspace=str(tmp_path), gemini_bin=str(fake_bin))
    agent.run("Question")
    assert str(fake_bin.parent) in captured_env.get("PATH", "").split(":")


def test_gemini_run_nonzero_exit_reads_error_object(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openjarvis.agents.gemini_subscription.is_gemini_subscription_available",
        lambda: True,
    )
    payload = json.dumps(
        {"error": {"message": "Manual authorization is required", "code": 41}}
    )

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=1, stdout=payload)

    monkeypatch.setattr("openjarvis.agents.gemini_subscription.subprocess.run", fake_run)
    agent = GeminiSubscriptionAgent(workspace=str(tmp_path), gemini_bin="/fake/bin/gemini")
    result = agent.run("Question")
    assert result.metadata["error"] is True
    assert "Manual authorization" in result.content
