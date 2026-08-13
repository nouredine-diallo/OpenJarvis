"""Tests for the GitHub artefact backup helper (permanent "Preuves" links
that survive the PC being off, since GitHub hosts the file, not the
local server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.tools import artifact_backup


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _isolated_repo_dir(tmp_path, monkeypatch):
    """Never let tests touch the real backup repo clone on disk."""
    monkeypatch.setattr(artifact_backup, "_REPO_DIR", tmp_path / "artifact_backup_repo")


def test_push_artifact_returns_permanent_url_on_success(tmp_path, monkeypatch):
    src = tmp_path / "final.png"
    src.write_bytes(b"\x89PNG fake")

    calls = []

    def _fake_run(args, *, timeout, cwd=None):
        calls.append(args)
        if args[:2] == ["gh", "repo"]:
            # Simulate the clone by creating the target dir + .git marker.
            (artifact_backup._REPO_DIR / ".git").mkdir(parents=True, exist_ok=True)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(artifact_backup, "_run", _fake_run)

    url = artifact_backup.push_artifact(str(src), "mission-123")

    assert url == (
        "https://github.com/nouredine-diallo/jarvis-artifacts/blob/main/"
        "missions/mission-123/final.png"
    )
    # The file was actually staged into the (fake) repo checkout.
    staged = artifact_backup._REPO_DIR / "missions" / "mission-123" / "final.png"
    assert staged.read_bytes() == b"\x89PNG fake"
    assert any(a[:2] == ["git", "push"] for a in calls)


def test_push_artifact_returns_none_when_missing_local_file():
    assert artifact_backup.push_artifact("/no/such/file.png", "mission-1") is None


def test_push_artifact_returns_none_when_clone_fails(tmp_path, monkeypatch):
    src = tmp_path / "final.png"
    src.write_bytes(b"data")

    def _fake_run(args, *, timeout, cwd=None):
        return _FakeCompletedProcess(returncode=1)

    monkeypatch.setattr(artifact_backup, "_run", _fake_run)

    assert artifact_backup.push_artifact(str(src), "mission-1") is None


def test_push_artifact_returns_none_when_push_fails(tmp_path, monkeypatch):
    src = tmp_path / "final.png"
    src.write_bytes(b"data")

    def _fake_run(args, *, timeout, cwd=None):
        if args[:2] == ["gh", "repo"]:
            (artifact_backup._REPO_DIR / ".git").mkdir(parents=True, exist_ok=True)
            return _FakeCompletedProcess(returncode=0)
        if args[:2] == ["git", "push"]:
            return _FakeCompletedProcess(returncode=1)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(artifact_backup, "_run", _fake_run)

    assert artifact_backup.push_artifact(str(src), "mission-1") is None


def test_push_artifact_tolerates_nothing_to_commit(tmp_path, monkeypatch):
    """Re-pushing the same artefact (e.g. a retry) must not be treated as
    a failure just because git has nothing new to commit."""
    src = tmp_path / "final.png"
    src.write_bytes(b"data")

    def _fake_run(args, *, timeout, cwd=None):
        if args[:2] == ["gh", "repo"]:
            (artifact_backup._REPO_DIR / ".git").mkdir(parents=True, exist_ok=True)
            return _FakeCompletedProcess(returncode=0)
        if args[:2] == ["git", "commit"]:
            return _FakeCompletedProcess(returncode=1, stdout=b"nothing to commit, working tree clean")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(artifact_backup, "_run", _fake_run)

    url = artifact_backup.push_artifact(str(src), "mission-1")
    assert url is not None


def test_push_artifact_swallows_unexpected_exceptions(tmp_path, monkeypatch):
    src = tmp_path / "final.png"
    src.write_bytes(b"data")

    def _fake_run(*args, **kwargs):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(artifact_backup, "_run", _fake_run)

    assert artifact_backup.push_artifact(str(src), "mission-1") is None
