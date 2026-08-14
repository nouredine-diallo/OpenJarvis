"""Tests for the cloud data backup (memory/history disaster recovery).

The git push itself is mocked (it's the same proven mechanism as
artifact_backup), but everything that could silently *corrupt or leak*
data is exercised for real: consistent SQLite snapshots of a live WAL-mode
database, gzip determinism, credential stripping, and a full
backup -> restore round-trip. An untested backup is not a backup.
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import pytest

from openjarvis.tools import data_backup
from openjarvis.tools.data_backup import (
    BACKUP_FILES,
    DataBackupService,
    backup_data,
    restore_from_backup,
    snapshot_file,
)


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_db(path: Path, rows: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")  # match production
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"row{i}",) for i in range(rows)])
    conn.commit()
    conn.close()


class TestAllowlistSafety:
    def test_secret_files_are_never_in_the_allowlist(self):
        """~/.openjarvis also holds cloud-keys.env -- a directory sweep
        would publish credentials to GitHub. This is the guard."""
        for dangerous in ("cloud-keys.env", ".env", "cloud-keys", ".venv"):
            assert dangerous not in BACKUP_FILES

    def test_text_files_are_credential_stripped(self, tmp_path: Path):
        src = tmp_path / "notes.md"
        src.write_text("note\nGH_TOKEN=ghp_" + "a" * 36 + "\nend")
        dest = tmp_path / "notes.md.gz"
        assert snapshot_file(src, dest) is True
        restored = gzip.open(dest, "rb").read().decode()
        assert "ghp_aaa" not in restored
        assert "REDACTED" in restored


class TestSnapshot:
    def test_sqlite_snapshot_is_valid_and_complete(self, tmp_path: Path):
        """A live WAL-mode db must snapshot to a consistent, readable copy
        with all committed rows -- a plain file copy can miss rows still
        living in the WAL."""
        src = tmp_path / "live.db"
        _make_db(src, rows=5)
        dest = tmp_path / "live.db.gz"
        assert snapshot_file(src, dest) is True

        restored = tmp_path / "restored.db"
        restored.write_bytes(gzip.open(dest, "rb").read())
        conn = sqlite3.connect(restored)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5
        conn.close()

    def test_snapshot_is_deterministic(self, tmp_path: Path):
        """Identical input must gzip to identical bytes, otherwise the
        skip-if-unchanged check never fires and every backup cycle commits
        a fresh multi-megabyte blob."""
        src = tmp_path / "live.db"
        _make_db(src)
        a, b = tmp_path / "a.gz", tmp_path / "b.gz"
        snapshot_file(src, a)
        snapshot_file(src, b)
        assert a.read_bytes() == b.read_bytes()

    def test_snapshot_never_mutates_the_source(self, tmp_path: Path):
        src = tmp_path / "live.db"
        _make_db(src)
        before = src.read_bytes()
        snapshot_file(src, tmp_path / "out.gz")
        assert src.read_bytes() == before

    def test_missing_source_fails_cleanly(self, tmp_path: Path):
        assert snapshot_file(tmp_path / "nope.db", tmp_path / "out.gz") is False


class TestBackupAndRestore:
    @pytest.fixture(autouse=True)
    def _isolated_repo(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.setattr(data_backup, "_REPO_DIR", repo)
        monkeypatch.setattr(data_backup, "_run", lambda *a, **k: _FakeCompletedProcess())
        return repo

    def test_round_trip_preserves_data(self, tmp_path: Path):
        """The guarantee that matters: back up, then restore onto a fresh
        machine and get usable data back."""
        source = tmp_path / "openjarvis"
        source.mkdir()
        _make_db(source / "memory_vec.db", rows=7)
        (source / "memory_facts.jsonl").write_text('{"text": "Prefers French", "kind": "preference"}\n')

        summary = backup_data(source_dir=source)
        assert summary["pushed"] is True
        assert "memory_vec.db" in summary["changed"]

        fresh = tmp_path / "fresh"
        restored = restore_from_backup(fresh)
        assert "memory_vec.db" in restored["restored"]

        conn = sqlite3.connect(fresh / "memory_vec.db")
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 7
        conn.close()
        assert "Prefers French" in (fresh / "memory_facts.jsonl").read_text()

    def test_second_backup_skips_unchanged_files(self, tmp_path: Path):
        source = tmp_path / "openjarvis"
        source.mkdir()
        _make_db(source / "memory_vec.db")

        first = backup_data(source_dir=source)
        assert "memory_vec.db" in first["changed"]

        second = backup_data(source_dir=source)
        assert second["changed"] == []
        assert any("inchangé" in s for s in second["skipped"])
        assert second["pushed"] is False

    def test_changed_file_is_backed_up_again(self, tmp_path: Path):
        source = tmp_path / "openjarvis"
        source.mkdir()
        facts = source / "memory_facts.jsonl"
        facts.write_text("first\n")
        backup_data(source_dir=source)

        facts.write_text("first\nsecond\n")
        again = backup_data(source_dir=source)
        assert "memory_facts.jsonl" in again["changed"]

    def test_absent_files_are_skipped_not_errors(self, tmp_path: Path):
        source = tmp_path / "openjarvis"
        source.mkdir()  # completely empty
        summary = backup_data(source_dir=source)
        assert summary["errors"] == []
        assert len(summary["skipped"]) == len(BACKUP_FILES)

    def test_restore_never_overwrites_existing_files(self, tmp_path: Path):
        source = tmp_path / "openjarvis"
        source.mkdir()
        (source / "memory_facts.jsonl").write_text("backed up\n")
        backup_data(source_dir=source)

        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "memory_facts.jsonl").write_text("LOCAL DATA\n")
        summary = restore_from_backup(dest)

        assert (dest / "memory_facts.jsonl").read_text() == "LOCAL DATA\n"
        assert any("déjà présent" in s for s in summary["skipped"])

    def test_push_failure_is_reported_not_raised(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "openjarvis"
        source.mkdir()
        (source / "memory_facts.jsonl").write_text("x\n")

        def _fail_on_push(args, **kwargs):
            if args[:2] == ["git", "push"]:
                return _FakeCompletedProcess(returncode=1, stderr=b"network down")
            return _FakeCompletedProcess()

        monkeypatch.setattr(data_backup, "_run", _fail_on_push)
        summary = backup_data(source_dir=source)
        assert summary["pushed"] is False
        assert summary["errors"]


class TestService:
    def test_stop_is_idempotent_and_bounded(self):
        svc = DataBackupService(interval_seconds=3600)
        svc.start()
        svc.stop()
        svc.stop()  # must not raise or hang

    def test_backup_failure_never_escapes_the_thread(self, monkeypatch):
        monkeypatch.setattr(
            data_backup, "backup_data", lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        svc = DataBackupService(interval_seconds=0.01)
        svc.start()
        svc.stop()  # the thread must have survived / not crashed the process
