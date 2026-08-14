"""GitHub repository search -- structured discovery of open-source projects.

Brique 3 (docs/SPEC_BRIQUE3_RECHERCHE.md, decision: dedicated tool). A
general web search returns blog posts *about* projects; this returns the
projects themselves with the fields that actually drive a choice (stars,
language, last push, license), which is a materially better answer to
"what open-source X exists?" than a noisy page of links.

Free and keyless. GitHub allows 10 unauthenticated search requests per
minute, so calls are paced process-wide to stay under it rather than
letting a research loop burn the quota and start failing mid-run.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

_API = "https://api.github.com/search/repositories"
#: 10 req/min unauthenticated -> 6s apart is the theoretical floor; 7s
#: leaves headroom for clock skew and concurrent callers.
_MIN_INTERVAL_S = 7.0
_TIMEOUT_S = 15.0

_rate_lock = threading.Lock()
_last_call_at = 0.0


def _pace() -> None:
    """Block just long enough to respect GitHub's unauthenticated rate
    limit. Process-wide, so several agents/tools sharing this tool can't
    collectively exceed it."""
    global _last_call_at
    with _rate_lock:
        elapsed = time.monotonic() - _last_call_at
        if elapsed < _MIN_INTERVAL_S and _last_call_at > 0.0:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        _last_call_at = time.monotonic()


@ToolRegistry.register("github_search")
class GitHubSearchTool(BaseTool):
    """Search GitHub repositories, ranked by stars."""

    tool_id = "github_search"
    is_local = False

    def __init__(self, token: str = "", max_results: int = 5) -> None:
        # A token only raises the rate limit; the tool is fully functional
        # without one, which keeps the zero-cost constraint intact.
        self._token = token or os.environ.get("GH_TOKEN", "")
        self._max_results = max_results

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="github_search",
            description=(
                "Search GitHub for open-source repositories matching a "
                "query, returned with stars, language, description, "
                "license and last-update date. Use this for discovery "
                "questions about software ('quelles alternatives open "
                "source à X', 'quel projet fait Y') -- it gives the "
                "projects themselves, where web_search would give blog "
                "posts about them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms, e.g. 'self-hosted AI assistant'.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language filter, e.g. 'python'.",
                    },
                    "max_results": {"type": "integer", "description": "Default 5."},
                },
                "required": ["query"],
            },
            category="search",
        )

    def execute(self, **params: Any) -> ToolResult:
        query = str(params.get("query", "") or "").strip()
        if not query:
            return ToolResult(tool_name=self.tool_id, content="No query provided.", success=False)

        language = str(params.get("language", "") or "").strip()
        if language:
            query = f"{query} language:{language}"
        max_results = int(params.get("max_results", self._max_results))

        try:
            import httpx
        except ImportError:
            return ToolResult(tool_name=self.tool_id, content="httpx n'est pas installé.", success=False)

        headers = {"accept": "application/vnd.github+json", "user-agent": "JARVIS-github-search"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"

        _pace()
        try:
            resp = httpx.get(
                _API,
                params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
                headers=headers,
                timeout=_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool_name=self.tool_id, content=f"Requête GitHub échouée : {exc}", success=False)

        if resp.status_code == 403:
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    "Limite de requêtes GitHub atteinte (10/min sans jeton). "
                    "Réessaie dans une minute."
                ),
                success=False,
            )
        if resp.status_code >= 300:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"GitHub a répondu {resp.status_code}: {resp.text[:200]}",
                success=False,
            )

        payload = resp.json()
        items: List[Dict[str, Any]] = payload.get("items", [])[:max_results]
        if not items:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Aucun dépôt trouvé pour : {query}",
                success=True,
                metadata={"total_count": payload.get("total_count", 0), "num_results": 0},
            )

        parts = []
        for it in items:
            license_info = (it.get("license") or {}).get("spdx_id") or "sans licence déclarée"
            parts.append(
                f"### {it.get('full_name')} — ⭐ {it.get('stargazers_count', 0):,}\n"
                f"Source: {it.get('html_url')}\n"
                f"Langage: {it.get('language') or 'n/a'} | Licence: {license_info} | "
                f"Dernière mise à jour: {(it.get('pushed_at') or '')[:10]}\n"
                f"Summary: {it.get('description') or '(pas de description)'}"
            )

        return ToolResult(
            tool_name=self.tool_id,
            content="\n\n---\n\n".join(parts),
            success=True,
            metadata={
                "total_count": payload.get("total_count", 0),
                "num_results": len(items),
                "authenticated": bool(self._token),
            },
        )


__all__ = ["GitHubSearchTool"]
