"""ClaudeSubscriptionAgent -- wraps the `claude` CLI in headless print mode.

Distinct from OpenJarvis's built-in ``claude_code`` agent (``claude_code.py``),
which drives the Claude Agent SDK over a Node.js subprocess and requires a
metered ``ANTHROPIC_API_KEY``. This wrapper shells out to the ``claude``
binary directly (``claude -p ... --output-format json``), authenticating
with whatever session is already logged into the CLI -- for a user on a
Claude.ai subscription (Pro/Max) that means usage counts against the
subscription's rate-limited quota, not a separate pay-per-token bill.

That quota is shared with the user's own interactive Claude Code sessions,
so this is the "frontier" tier of JARVIS's model router: reserved for steps
where quality matters more than speed or cost (hard coding, self-review,
planning) -- not the default for every mission step. See
``missions/engine.py`` (``coding_pr_steps``) for where it is actually used,
and ``MissionsConfig.claude_subscription_daily_budget_usd`` for the guard
that keeps automated missions from eating the quota the user needs for
their own manual work.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.engine._stubs import InferenceEngine

logger = logging.getLogger(__name__)


def find_claude_bin() -> str:
    """Locate the ``claude`` binary, including non-PATH installs (systemd
    user services run with a minimal PATH -- mirrors ``opencode.py``).

    Claude Code is commonly installed via ``npm i -g`` under a
    version-manager-owned prefix (nvm, volta, asdf, …), which systemd's
    default minimal PATH does not include -- verified empirically: a
    plain ``env -i PATH=/usr/bin:/bin which claude`` finds nothing even
    when an interactive shell resolves it fine via nvm's per-version bin
    dir. Glob those layouts explicitly rather than assuming a stable path.
    """
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "claude",
        home / ".claude" / "local" / "claude",
        home / ".npm-global" / "bin" / "claude",
    ]
    for pattern in (
        ".nvm/versions/node/*/bin/claude",
        ".volta/bin/claude",
        ".asdf/installs/nodejs/*/bin/claude",
    ):
        candidates.extend(sorted(home.glob(pattern), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "claude"


def is_claude_subscription_available() -> bool:
    """True if the ``claude`` binary is reachable.

    Does not check login state -- a stale/missing OAuth session still
    passes this check and fails at call time with a clear error instead.
    """
    bin_path = find_claude_bin()
    return shutil.which(bin_path) is not None or Path(bin_path).exists()


class ClaudeSubscriptionBudgetExceeded(RuntimeError):
    """Raised when the configured daily subscription-usage guard trips."""


@AgentRegistry.register("claude_subscription")
class ClaudeSubscriptionAgent(BaseAgent):
    """Headless coding/reasoning agent backed by the user's Claude subscription.

    Each call spawns ``claude -p <prompt> --output-format json
    --permission-mode acceptEdits`` in ``workspace`` and parses the JSON
    result. ``acceptEdits`` auto-accepts file edits *and* lets Bash tool
    calls (e.g. ``git diff``, running tests) proceed without a TTY prompt --
    verified empirically; the stricter default permission mode would hang
    waiting for interactive confirmation that never comes in a headless
    subprocess.
    """

    agent_id = "claude_subscription"
    accepts_tools = False
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: Optional[InferenceEngine] = None,
        model: str = "claude-sonnet-5",
        *,
        bus: Optional[EventBus] = None,
        workspace: str = "",
        permission_mode: str = "acceptEdits",
        timeout: int = 300,
        claude_bin: str = "",
        daily_budget_usd: float = 0.0,
        usage_log_path: str = "",
    ) -> None:
        super().__init__(engine, model, bus=bus)
        self._workspace = workspace or os.getcwd()
        self._permission_mode = permission_mode
        self._timeout = timeout
        self._claude_bin = claude_bin or find_claude_bin()
        # 0 = unmetered (still capped by the CLI's own subscription rate
        # limits, just not double-guarded here). Set > 0 to cap the
        # *estimated* daily spend this agent reports back to itself, so
        # automated missions cannot silently exhaust the quota the user
        # needs for their own interactive sessions.
        self._daily_budget_usd = daily_budget_usd
        self._usage_log_path = Path(
            usage_log_path or (Path.home() / ".openjarvis" / "claude_subscription_usage.jsonl")
        )

    # ------------------------------------------------------------------
    # Budget guard
    # ------------------------------------------------------------------

    def _today_spend(self) -> float:
        if not self._usage_log_path.exists():
            return 0.0
        cutoff = time.time() - 86400
        total = 0.0
        try:
            with self._usage_log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("ts", 0) >= cutoff:
                        total += float(rec.get("cost_usd", 0.0))
        except OSError:
            return 0.0
        return total

    def _log_spend(self, cost_usd: float) -> None:
        try:
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "cost_usd": cost_usd}) + "\n")
        except OSError:
            logger.debug("Failed to log claude_subscription usage", exc_info=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        if not is_claude_subscription_available():
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=(
                    "ClaudeSubscriptionAgent requires the 'claude' CLI "
                    "(Claude Code) on PATH, logged in via `claude login` "
                    "(Pro/Max subscription, not an API key)."
                ),
                turns=1,
                metadata={"error": True},
            )

        if self._daily_budget_usd > 0:
            spent = self._today_spend()
            if spent >= self._daily_budget_usd:
                self._emit_turn_end(turns=1, error=True)
                return AgentResult(
                    content=(
                        f"Claude subscription daily guard reached "
                        f"(${spent:.2f} >= ${self._daily_budget_usd:.2f} estimated over "
                        "24h) -- falling back rather than risk eating the quota "
                        "needed for interactive use."
                    ),
                    turns=1,
                    metadata={"error": True, "budget_exceeded": True},
                )

        cmd = [
            self._claude_bin,
            "-p",
            input,
            "--output-format",
            "json",
            "--permission-mode",
            self._permission_mode,
            # Without this, tool access defaults to whatever directories are
            # already trusted in this machine's ~/.claude config (observed
            # empirically: a call with cwd=<fresh scratch dir> still had its
            # Bash/git tool calls scoped to an unrelated, previously-trusted
            # project directory) -- explicit --add-dir is what actually
            # grants access to the mission's real target repo.
            "--add-dir",
            self._workspace,
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
                content=f"Claude subscription agent timed out after {self._timeout}s.",
                turns=1,
                metadata={"error": True, "error_type": "timeout"},
            )
        except FileNotFoundError:
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content="claude CLI not found at " + self._claude_bin,
                turns=1,
                metadata={"error": True},
            )

        if proc.returncode != 0:
            # `claude -p ... --output-format json` prints its error object to
            # stdout even on a non-zero exit (observed live: a same-day
            # failure logged "Unknown error" with empty stderr because this
            # branch never looked at stdout). Prefer that structured detail.
            detail = ""
            try:
                err_data = json.loads(proc.stdout) if proc.stdout else {}
                err_obj = err_data.get("error") if isinstance(err_data, dict) else None
                if isinstance(err_obj, dict):
                    detail = str(err_obj.get("message") or err_obj.get("type") or "")
                elif err_obj:
                    detail = str(err_obj)
            except json.JSONDecodeError:
                pass
            if not detail:
                detail = (proc.stderr or "").strip()[:500]
            if not detail:
                detail = f"Unknown error (exit code {proc.returncode}, no stdout/stderr)"
            logger.error("claude CLI exited with code %d: %s", proc.returncode, detail)
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=f"Claude subscription agent failed: {detail}",
                turns=1,
                metadata={"error": True, "returncode": proc.returncode},
            )

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            self._emit_turn_end(turns=1, error=True)
            return AgentResult(
                content=(proc.stdout or "").strip() or "Empty response from claude CLI",
                turns=1,
                metadata={"parse_error": True},
            )

        content = str(data.get("result", "") or "")
        cost = float(data.get("total_cost_usd", 0.0) or 0.0)
        self._log_spend(cost)

        self._emit_turn_end(turns=int(data.get("num_turns", 1) or 1))
        return AgentResult(
            content=content,
            turns=int(data.get("num_turns", 1) or 1),
            metadata={
                "session_id": data.get("session_id", ""),
                "duration_ms": data.get("duration_ms", 0),
                "subscription_cost_estimate_usd": cost,
                "is_error": bool(data.get("is_error", False)),
                "model": data.get("modelUsage", {}),
            },
        )


__all__ = [
    "ClaudeSubscriptionAgent",
    "ClaudeSubscriptionBudgetExceeded",
    "is_claude_subscription_available",
    "find_claude_bin",
]
