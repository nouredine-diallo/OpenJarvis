"""Best-effort liveness ping to the JARVIS control plane (Cloudflare
Worker, Phase A of removing the PC-dependency -- see PROJET_JARVIS.md).

Lets the always-on Worker know the PC is available to claim missions, so
it can route work to the PC (fast) instead of a cloud fallback worker
when the PC is on, and stop routing to the PC the moment it goes quiet.
Runs in a background thread; a failure here (network down, Worker
unreachable) must never affect the mission engine or the Telegram bot --
it only means the control plane will, correctly, start treating the PC
as offline after the freshness window elapses.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def send_heartbeat(
    control_plane_url: str,
    shared_secret: str,
    capabilities: Iterable[str],
    *,
    worker_id: str = "pc",
    timeout: float = 5.0,
) -> bool:
    """Single best-effort heartbeat POST. Returns True on HTTP 2xx."""
    url = control_plane_url.rstrip("/") + "/heartbeat"
    body = json.dumps({"worker_id": worker_id, "capabilities": list(capabilities)}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-control-plane-secret": shared_secret,
            # Cloudflare's edge 403s the default "Python-urllib/x.y" UA as
            # anti-abuse (found live: curl with that exact UA string also
            # gets 403, a normal one gets 200) -- independent of our Worker
            # code, so this has to be set here, not fixed on the Worker side.
            "user-agent": "JARVIS-PC-Worker/1.0 (+control-plane-heartbeat)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        logger.debug("Control plane heartbeat failed", exc_info=True)
        return False


def sync_memory(
    control_plane_url: str,
    shared_secret: str,
    *,
    facts_path: str = "",
    timeout: float = 15.0,
) -> int:
    """Push the local fact store to the control plane so the cloud path
    knows the user when the PC is off.

    Without this the offline assistant answers as a stranger -- it is what
    turns "PC éteint" from a downgrade into a continuation of the same
    JARVIS. Best-effort: returns the number of facts sent, 0 on any
    failure, never raises.
    """
    path = Path(facts_path) if facts_path else Path.home() / ".openjarvis" / "memory_facts.jsonl"
    if not path.exists():
        return 0
    try:
        facts: List[Dict[str, Any]] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = (entry.get("text") or "").strip()
            if text:
                facts.append({"id": str(i), "kind": entry.get("kind", "fact"), "text": text})
    except Exception:  # noqa: BLE001
        logger.debug("Could not read fact store for cloud sync", exc_info=True)
        return 0

    request = urllib.request.Request(
        control_plane_url.rstrip("/") + "/memory/sync",
        data=json.dumps({"facts": facts}).encode(),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-control-plane-secret": shared_secret,
            "user-agent": "JARVIS-PC-Worker/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return len(facts)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        logger.debug("Cloud memory sync failed", exc_info=True)
    return 0


class ControlPlaneHeartbeat:
    """Background thread sending periodic heartbeats. Never raises out of
    the thread; stop() is idempotent and joins with a short timeout."""

    def __init__(
        self,
        control_plane_url: str,
        shared_secret: str,
        capabilities: Iterable[str],
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        self._url = control_plane_url
        self._secret = shared_secret
        self._capabilities = list(capabilities)
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._url or not self._secret:
            logger.info("Control plane heartbeat disabled: URL or secret not configured")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="control-plane-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Memory is synced far less often than the heartbeat: the fact
        # store changes slowly, and re-uploading it every 30s would be
        # pure waste. Every ~5 minutes keeps the offline assistant
        # current without the churn.
        beats_per_sync = max(1, int(300 / max(self._interval, 1)))
        beat = 0
        while not self._stop.is_set():
            send_heartbeat(self._url, self._secret, self._capabilities)
            if beat % beats_per_sync == 0:
                count = sync_memory(self._url, self._secret)
                if count:
                    logger.debug("Synced %d facts to the control plane", count)
            beat += 1
            self._stop.wait(self._interval)
