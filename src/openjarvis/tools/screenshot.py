"""Visual proof pipeline: screenshot a running dev server, if one exists.

Design (2026-08-13, confirmed with the user before building):
- One capture of the *current/final* state, not a before/after pair --
  delivered as a single Telegram photo.
- Automatic "when possible" after a coding mission (Setup/Implement/Test/
  Review/Ship): silently skipped if no web server is detected, never
  forces a screenshot on a script/library project that has no UI.
- Also available on demand (see ``tools/screenshot.py``'s
  ``ScreenshotTool``) so the user can ask "montre-moi l'image" /
  "quel est l'état actuel" outside of a mission.
- RAM-guarded: this PC hit OOM twice today on far smaller local-model
  workloads (see PLAN.md D9) -- a headless Chromium is much lighter
  (~150-300 MB) but still checked before launch, never launched blind.
- Config-gated (``missions.enable_visual_proof``) so it can be switched
  off entirely if the user doesn't want the extra work/RAM per mission.
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ports dev servers commonly bind by convention (Next.js/CRA/Vite/Django/
# Flask/Streamlit/...). Checked cheapest-first: a raw TCP connect (a few ms)
# before ever trying an HTTP request.
_CANDIDATE_PORTS = (3000, 5173, 8080, 5000, 4321, 8501, 4200, 8000, 3001, 5001)
# 8000 is JARVIS's own port -- included so a project that happens to also
# pick 8000 elsewhere isn't silently missed, but see find_dev_server_url:
# it's explicitly excluded to avoid screenshotting JARVIS's own API.

_JARVIS_OWN_PORT = 8000


def ram_available_mb() -> float:
    """Available RAM in MB, read directly from /proc/meminfo (MemAvailable
    -- the kernel's own "can allocate this much without swapping heavily"
    estimate, not just free-page count). Returns a large number (never
    blocks a capture) if unreadable -- the guard degrades to "don't guard"
    rather than "never capture" on an unexpected platform.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (OSError, ValueError, IndexError):
        logger.debug("Could not read /proc/meminfo", exc_info=True)
    return float("inf")


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_dev_server_url(extra_text: str = "") -> Optional[str]:
    """Best-effort detection of a locally running dev server.

    Two passes: first look for an explicit "http://localhost:PORT" or
    "http://127.0.0.1:PORT" printed in ``extra_text`` (mission step
    results often contain the dev server's own startup banner, e.g. "Local:
    http://localhost:3000" -- far more reliable than guessing), then fall
    back to probing the common ports directly. Returns ``None`` (never
    raises) if nothing responds -- the caller treats that as "not
    applicable here", not an error.
    """
    import re

    for match in re.finditer(
        r"https?://(?:localhost|127\.0\.0\.1)(?::(\d{2,5}))?", extra_text or ""
    ):
        port = int(match.group(1)) if match.group(1) else 80
        if port == _JARVIS_OWN_PORT:
            continue
        if _port_open(port):
            return f"http://127.0.0.1:{port}"

    for port in _CANDIDATE_PORTS:
        if port == _JARVIS_OWN_PORT:
            continue
        if _port_open(port):
            return f"http://127.0.0.1:{port}"
    return None


def capture_screenshot(
    url: str,
    out_path: str,
    *,
    min_ram_mb: float = 1024.0,
    nav_timeout_ms: int = 15000,
) -> tuple[Optional[str], str]:
    """Capture a full-page screenshot of ``url`` to ``out_path``.

    Returns ``(path_or_None, reason)``: ``reason`` is always a short
    human-readable string (why it was skipped, or "ok"). Never raises --
    every failure mode (RAM too tight, playwright missing, browser launch
    failure, navigation timeout) is caught and reported, not propagated,
    so a screenshot attempt can never itself fail a mission step.
    """
    available = ram_available_mb()
    if available < min_ram_mb:
        return None, (
            f"RAM disponible trop juste pour un navigateur headless "
            f"({available:.0f} Mo < {min_ram_mb:.0f} Mo requis) -- capture ignorée."
        )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright n'est pas installé (uv sync --extra browser)."

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                page.goto(url, timeout=nav_timeout_ms, wait_until="load")
                page.screenshot(path=out_path, full_page=True)
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Screenshot capture failed for %s", url, exc_info=True)
        return None, f"échec de capture ({type(exc).__name__}): {str(exc)[:200]}"

    duration = time.monotonic() - t0
    return out_path, f"ok ({duration:.1f}s)"


__all__ = [
    "ram_available_mb",
    "find_dev_server_url",
    "capture_screenshot",
]
