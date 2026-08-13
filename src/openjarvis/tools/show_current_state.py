"""On-demand visual proof: "montre-moi l'image" / "quel est l'état actuel".

Confirmed with the user (2026-08-13): the default behaviour is one
screenshot of whatever's currently running, sent as one photo -- available
both automatically after a coding mission (see missions/engine.py's
_finish hook) AND on demand through this tool, so a plain "montre-moi
l'image" mid-conversation works too, not just right after a mission.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.ssrf import check_ssrf
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.screenshot import capture_screenshot, find_dev_server_url

logger = logging.getLogger(__name__)


@ToolRegistry.register("show_current_state")
class ShowCurrentStateTool(BaseTool):
    """Screenshot whatever local dev server is currently running."""

    tool_id = "show_current_state"
    # A photo delivery hook -- same (target, path, caption) -> bool shape
    # as MissionEngine's photo_sender (serve.py's _mission_photo_sender is
    # reused directly for both) -- posts the image where the tool result
    # alone cannot, since tool results are text. Set post-build, same
    # pattern as ``LaunchMissionTool._mission_engine``.
    _photo_sender: Optional[Any] = None
    _min_ram_mb: float = 1024.0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="show_current_state",
            description=(
                "Take a screenshot of the app/site currently running "
                "locally (auto-detects a dev server on common ports) and "
                "send it as a photo. Use this when the user asks to see "
                "the current state, 'montre-moi l'image', 'montre-moi le "
                "résultat', or similar -- not for missions that already "
                "finished (those get a photo automatically already)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Optional explicit URL to screenshot. Omit to "
                            "auto-detect a locally running dev server."
                        ),
                    },
                },
                "required": [],
            },
            category="visual",
        )

    def execute(self, **params: Any) -> ToolResult:
        url = str(params.get("url", "") or "").strip()
        if url:
            err = check_ssrf(url)
            if err:
                return ToolResult(
                    tool_name=self.tool_id,
                    content=f"URL refusée ({err}).",
                    success=False,
                )
        else:
            url = find_dev_server_url()
            if not url:
                return ToolResult(
                    tool_name=self.tool_id,
                    content=(
                        "Aucun serveur local détecté (ports courants "
                        "vérifiés : 3000, 5173, 8080, 5000, ...). Si "
                        "l'application tourne sur un autre port, précise "
                        "l'URL."
                    ),
                    success=False,
                )

        out_dir = get_config_dir() / "mission_artifacts" / "on_demand"
        out_path = str(out_dir / f"shot_{int(time.time())}.png")
        path, reason = capture_screenshot(url, out_path, min_ram_mb=self._min_ram_mb)

        if not path:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Capture impossible : {reason}",
                success=False,
            )

        sent = False
        target = str(params.get("_channel", "") or params.get("requested_by", "") or "")
        if self._photo_sender is not None and target:
            try:
                sent = bool(self._photo_sender(target, path, f"État actuel — {url}"))
            except Exception:  # noqa: BLE001
                logger.debug("Photo send failed", exc_info=True)

        return ToolResult(
            tool_name=self.tool_id,
            content=(
                f"Capture de {url} réussie ({reason})."
                + (" Envoyée en photo." if sent else f" Fichier : {path}")
            ),
            metadata={"path": path, "url": url, "sent": sent},
        )


__all__ = ["ShowCurrentStateTool"]
