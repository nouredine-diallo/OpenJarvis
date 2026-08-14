"""Cloud backup of JARVIS's irreplaceable local data.

The engine's *code* has been safe on GitHub since 2026-08-13, but its
*data* -- the memory built by Brique 2, the fact store, mission and
conversation history -- lived only on this PC's disk. That was the last
remaining single point of failure with no protection at all: a dead disk
meant losing everything JARVIS had learned, permanently. This module
closes it, reusing the same proven mechanism as the screenshot artifact
backup (a private GitHub repo, zero cost, no new service to configure).

Three things this gets right that a naive `cp` to a repo would not:

1. **Consistent SQLite snapshots.** These databases run in WAL mode (see
   the ``.db-wal``/``.db-shm`` files next to them), so copying the ``.db``
   file alone can capture a torn, partially-committed state -- committed
   rows living in the WAL would simply be missing. ``VACUUM INTO`` asks
   SQLite itself for an atomic, self-contained snapshot instead
   (measured: 0.5s for the 3.1 MB vector store, ``integrity_check`` ok).

2. **An explicit allowlist, never a directory sweep.** ``~/.openjarvis``
   also contains ``cloud-keys.env`` and a whole ``.venv`` -- a
   "back up the folder" approach would push credentials to GitHub. Only
   the files named in :data:`BACKUP_FILES` are ever touched, and text
   files additionally go through the credential stripper on the way out.

3. **Content-addressed skipping.** Snapshots are only committed when
   their content actually changed, so an idle JARVIS doesn't accumulate
   an identical multi-megabyte blob in git history every hour.

Restore is a first-class function here (:func:`restore_from_backup`), not
an afterthought -- an untested backup is not a backup.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_REPO_DIR = Path.home() / ".openjarvis" / "artifact_backup_repo"
_REPO_SLUG = "nouredine-diallo/jarvis-artifacts"
_BACKUP_PREFIX = "data"

#: Files worth protecting, relative to ~/.openjarvis. Deliberately an
#: explicit list (see module docstring): everything here is either
#: irreplaceable user data or cheap-but-useful config. Regenerable or
#: sensitive files (telemetry/traces/audit dbs, cloud-keys.env, .venv,
#: logs) are intentionally absent.
BACKUP_FILES: Tuple[str, ...] = (
    "memory_facts.jsonl",  # the fact store -- most precious, human-readable
    "memory.db",  # sparse/FTS5 retrieval index
    "memory_vec.db",  # dense/sqlite-vec embeddings (Brique 2)
    "missions.db",  # mission history
    "sessions.db",  # conversation history
    "knowledge.db",  # knowledge graph
    "MEMORY.md",
    "SOUL.md",
    "USER.md",
    "config.toml",  # no secrets (those live in JARVIS/.env), useful to restore
)

_TEXT_SUFFIXES = {".md", ".toml", ".jsonl", ".json", ".yaml", ".yml"}


def _run(args: List[str], *, timeout: float, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd) if cwd else None, capture_output=True, timeout=timeout
    )


def _ensure_repo_cloned() -> bool:
    if (_REPO_DIR / ".git").exists():
        return True
    try:
        _REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        result = _run(["gh", "repo", "clone", _REPO_SLUG, str(_REPO_DIR)], timeout=30)
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        logger.debug("data backup: clone failed", exc_info=True)
        return False


def snapshot_file(src: Path, dest: Path) -> bool:
    """Write a consistent, gzipped snapshot of *src* to *dest*.

    SQLite files go through ``VACUUM INTO`` (atomic, WAL-safe -- see the
    module docstring); everything else is copied. Text files are
    credential-stripped first, so a secret that ever lands in a config or
    a memory note is redacted rather than published.

    Returns True on success. Never raises.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".db":
            with tempfile.TemporaryDirectory() as tmp:
                snap = Path(tmp) / "snap.db"
                # Read-only URI: never let a backup mutate the live db.
                conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
                try:
                    conn.execute("VACUUM INTO ?", (str(snap),))
                finally:
                    conn.close()
                raw = snap.read_bytes()
        else:
            raw = src.read_bytes()
            if src.suffix in _TEXT_SUFFIXES:
                from openjarvis.security.credential_stripper import CredentialStripper

                text = raw.decode("utf-8", errors="replace")
                raw = CredentialStripper().strip(text).encode("utf-8")

        # mtime=0 so an unchanged file always gzips to identical bytes --
        # otherwise the embedded timestamp would defeat the "skip if
        # unchanged" check below and commit a new blob every single run.
        # (GzipFile does not close a fileobj it was handed, hence the
        # explicit outer `with`.)
        with dest.open("wb") as raw_out:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as gz:
                gz.write(raw)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("data backup: snapshot of %s failed", src, exc_info=True)
        return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_data(*, source_dir: Optional[Path] = None) -> Dict[str, object]:
    """Snapshot every file in :data:`BACKUP_FILES` and push the changed
    ones to the private backup repo.

    Returns a summary dict (``pushed``, ``changed``, ``skipped``,
    ``errors``). Never raises -- backup is best-effort and must never
    disturb the running assistant.
    """
    base = source_dir or (Path.home() / ".openjarvis")
    summary: Dict[str, object] = {
        "pushed": False,
        "changed": [],
        "skipped": [],
        "errors": [],
    }

    with _LOCK:
        if not _ensure_repo_cloned():
            summary["errors"].append("clone failed")  # type: ignore[union-attr]
            return summary

        _run(["git", "pull", "--ff-only"], timeout=30, cwd=_REPO_DIR)

        changed: List[str] = []
        for name in BACKUP_FILES:
            src = base / name
            if not src.exists():
                summary["skipped"].append(f"{name} (absent)")  # type: ignore[union-attr]
                continue

            rel = f"{_BACKUP_PREFIX}/{name}.gz"
            dest = _REPO_DIR / rel
            previous = _sha256(dest) if dest.exists() else ""

            if not snapshot_file(src, dest):
                summary["errors"].append(f"{name} (snapshot failed)")  # type: ignore[union-attr]
                continue

            if _sha256(dest) == previous:
                summary["skipped"].append(f"{name} (inchangé)")  # type: ignore[union-attr]
                continue

            _run(["git", "add", rel], timeout=15, cwd=_REPO_DIR)
            changed.append(name)

        summary["changed"] = changed
        if not changed:
            return summary

        commit = _run(
            ["git", "commit", "-m", f"data backup: {', '.join(changed)}"],
            timeout=15,
            cwd=_REPO_DIR,
        )
        if commit.returncode != 0 and b"nothing to commit" not in commit.stdout:
            summary["errors"].append("commit failed")  # type: ignore[union-attr]
            return summary

        push = _run(["git", "push"], timeout=60, cwd=_REPO_DIR)
        if push.returncode != 0:
            summary["errors"].append(f"push failed: {push.stderr[:200]!r}")  # type: ignore[union-attr]
            return summary

        summary["pushed"] = True
        return summary


def restore_from_backup(dest_dir: Path, *, files: Optional[List[str]] = None) -> Dict[str, object]:
    """Restore backed-up data into *dest_dir* (e.g. a fresh machine's
    ``~/.openjarvis``).

    Deliberately never overwrites an existing file: restoring onto a live
    install should be a conscious act, not something a stray call can do.
    Existing files are reported as skipped so the caller can decide.
    """
    summary: Dict[str, object] = {"restored": [], "skipped": [], "errors": []}

    with _LOCK:
        if not _ensure_repo_cloned():
            summary["errors"].append("clone failed")  # type: ignore[union-attr]
            return summary
        _run(["git", "pull", "--ff-only"], timeout=30, cwd=_REPO_DIR)

        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in files or BACKUP_FILES:
            src = _REPO_DIR / f"{_BACKUP_PREFIX}/{name}.gz"
            if not src.exists():
                summary["skipped"].append(f"{name} (pas de sauvegarde)")  # type: ignore[union-attr]
                continue
            target = dest_dir / name
            if target.exists():
                summary["skipped"].append(f"{name} (déjà présent)")  # type: ignore[union-attr]
                continue
            try:
                with gzip.open(src, "rb") as gz:
                    target.write_bytes(gz.read())
                summary["restored"].append(name)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append(f"{name}: {exc}")  # type: ignore[union-attr]

    return summary


class DataBackupService:
    """Periodic background data backup. Mirrors ControlPlaneHeartbeat's
    shape (daemon thread, idempotent stop, never raises out of the
    thread) so the lifecycle is familiar and equally unobtrusive."""

    def __init__(self, *, interval_seconds: float = 3600.0) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="data-backup")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # An initial delay keeps startup light: a backup in the first
        # seconds of boot competes with model loading and the first
        # user request for both CPU and RAM.
        if self._stop.wait(60.0):
            return
        while not self._stop.is_set():
            try:
                summary = backup_data()
                if summary.get("pushed"):
                    logger.info("Data backup pushed: %s", summary.get("changed"))
            except Exception:  # noqa: BLE001
                logger.debug("Data backup cycle failed", exc_info=True)
            self._stop.wait(self._interval)


__all__ = [
    "BACKUP_FILES",
    "DataBackupService",
    "backup_data",
    "restore_from_backup",
    "snapshot_file",
]
