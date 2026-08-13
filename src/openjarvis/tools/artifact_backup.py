"""Best-effort backup of mission artefacts (screenshots) to a private
GitHub repo, so "Preuves" links in mission reports stay reachable even
when the PC's tunnel isn't running -- or the PC itself is off. GitHub
hosts the file, not the local server, so the link survives independently
of the machine that produced it. Never raises: any failure here must not
block or fail the mission itself.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_REPO_DIR = Path.home() / ".openjarvis" / "artifact_backup_repo"
_REPO_SLUG = "nouredine-diallo/jarvis-artifacts"


def _run(args: list[str], *, timeout: float, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd) if cwd else None, capture_output=True, timeout=timeout
    )


def _ensure_repo_cloned() -> bool:
    if (_REPO_DIR / ".git").exists():
        return True
    try:
        _REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        result = _run(
            ["gh", "repo", "clone", _REPO_SLUG, str(_REPO_DIR)], timeout=20
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        logger.debug("artifact backup: clone failed", exc_info=True)
        return False


def push_artifact(local_path: str, mission_id: str) -> Optional[str]:
    """Copy local_path into the artefacts repo under missions/<id>/<name>,
    commit and push it. Returns the permanent github.com blob URL on
    success, None on any failure -- callers must treat this as best-effort
    and keep using the local artefact path if it returns None."""
    src = Path(local_path)
    if not src.exists():
        return None
    with _LOCK:
        try:
            if not _ensure_repo_cloned():
                return None
            _run(["git", "pull", "--ff-only"], timeout=15, cwd=_REPO_DIR)

            dest_rel = f"missions/{mission_id}/{src.name}"
            dest = _REPO_DIR / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

            add = _run(["git", "add", dest_rel], timeout=10, cwd=_REPO_DIR)
            if add.returncode != 0:
                return None

            commit = _run(
                ["git", "commit", "-m", f"artefact: mission {mission_id}"],
                timeout=10,
                cwd=_REPO_DIR,
            )
            if commit.returncode != 0 and b"nothing to commit" not in commit.stdout:
                return None

            push = _run(["git", "push"], timeout=20, cwd=_REPO_DIR)
            if push.returncode != 0:
                return None

            return f"https://github.com/{_REPO_SLUG}/blob/main/{dest_rel}"
        except Exception:  # noqa: BLE001
            logger.debug("artifact backup: push failed", exc_info=True)
            return None
