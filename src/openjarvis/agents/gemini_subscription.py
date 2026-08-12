"""GeminiSubscriptionAgent -- wraps the official `gemini` CLI in headless mode.

Same pattern as ``claude_subscription.py``: shells out to a vendor CLI that
authenticates via an OAuth-backed personal account subscription (Google
Gemini Pro/Ultra, via ``google-gemini/gemini-cli``) instead of a metered
``GEMINI_API_KEY``. Distinct from ``agents/opencode.py``'s Gemini path,
which requires ``GOOGLE_GENERATIVE_AI_API_KEY`` (metered) and was found to
return empty responses in testing -- this uses Google's own official CLI
instead of routing through opencode's generic provider layer.

Note: Antigravity IDE (also Google, also OAuth-backed) stores its own
token under ``~/.gemini/antigravity-cli/`` -- verified empirically that the
standalone `gemini` CLI does *not* automatically reuse it; it needs its own
one-time interactive login (``gemini`` run once in a real terminal, choose
"Login with Google"). Until that happens, every call here fails cleanly
with ``metadata={"error": True}`` and the caller (MissionEngine) falls
back to the next configured tier -- no code change needed once the user
completes that one-time login.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)


def find_gemini_bin() -> str:
    """Locate the `gemini` binary, including version-manager installs
    (see claude_subscription.find_claude_bin -- same rationale: a systemd
    service's minimal PATH does not see nvm/volta/asdf-managed npm bins)."""
    found = shutil.which("gemini")
    if found:
        return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "gemini",
        home / ".npm-global" / "bin" / "gemini",
    ]
    for pattern in (
        ".nvm/versions/node/*/bin/gemini",
        ".volta/bin/gemini",
        ".asdf/installs/nodejs/*/bin/gemini",
    ):
        candidates.extend(sorted(home.glob(pattern), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "gemini"


def is_gemini_subscription_available() -> bool:
    """True if the `gemini` binary is reachable. Does not check login
    state -- a missing OAuth session still passes this and fails at call
    time with a clear, gracefully-handled error instead."""
    bin_path = find_gemini_bin()
    return shutil.which(bin_path) is not None or Path(bin_path).exists()


@AgentRegistry.register("gemini_subscription")
class GeminiSubscriptionAgent(BaseAgent):
    """Headless coding/reasoning agent backed by a Gemini Pro/Ultra subscription.

    Each call spawns ``gemini --skip-trust --approval-mode auto_edit
    --include-directories <workspace> -o json -p <prompt>``. ``auto_edit``
    auto-approves file edits; unlike Claude's ``acceptEdits``, Bash/shell
    tool calls under Gemini CLI may still prompt -- if that turns out to
    require ``--approval-mode yolo`` in practice, that is a one-line change
    here once verified against a logged-in session.
    """

    agent_id = "gemini_subscription"
    accepts_tools = False
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: Optional[InferenceEngine] = None,
        model: str = "gemini",
        *,
        bus: Optional[EventBus] = None,
        workspace: str = "",
        approval_mode: str = "auto_edit",
        timeout: int = 300,
        gemini_bin: str = "",
    ) -> None:
        super().__init__(engine, model, bus=bus)
        self._workspace = workspace or os.getcwd()
        self._approval_mode = approval_mode
        self._timeout = timeout
        self._gemini_bin = gemini_bin or find_gemini_bin()

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        if not is_gemini_subscription_available():
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=(
                    "GeminiSubscriptionAgent requires the 'gemini' CLI "
                    "(npm i -g @google/gemini-cli), logged in via a "
                    "personal Google account (run `gemini` once "
                    "interactively -> Login with Google), not an API key."
                ),
                turns=1,
                metadata={"error": True},
            )

        cmd = [
            self._gemini_bin,
            "--skip-trust",
            "--approval-mode",
            self._approval_mode,
            "--include-directories",
            self._workspace,
            "-o",
            "json",
            "-p",
            input,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=f"Gemini subscription agent timed out after {self._timeout}s.",
                turns=1,
                metadata={"error": True, "error_type": "timeout"},
            )
        except FileNotFoundError:
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content="gemini CLI not found at " + self._gemini_bin,
                turns=1,
                metadata={"error": True},
            )

        raw = (proc.stdout or "").strip()
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {}

        if data.get("error") or (proc.returncode != 0 and not data.get("response")):
            err = data.get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
            msg = msg or (proc.stderr or "").strip()[:500] or "Unknown error"
            logger.error("gemini CLI failed: %s", msg)
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=f"Gemini subscription agent failed: {msg}",
                turns=1,
                metadata={"error": True, "returncode": proc.returncode},
            )

        content = str(data.get("response", "") or raw)
        if not content.strip():
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content="Empty response from gemini CLI",
                turns=1,
                metadata={"error": True},
            )

        self._emit_turn_end(turns=1)
        return AgentResult(
            content=content,
            turns=1,
            metadata={"stats": data.get("stats", {})},
        )


__all__ = [
    "GeminiSubscriptionAgent",
    "is_gemini_subscription_available",
    "find_gemini_bin",
]
