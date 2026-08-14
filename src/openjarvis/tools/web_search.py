"""Web search tool — Tavily API with DuckDuckGo fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.ssrf import check_ssrf
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

#: Forced DDG locale -- see _duckduckgo_search.
_DDG_REGION = "fr-fr"
_SEARXNG_TIMEOUT_S = 15.0


@ToolRegistry.register("web_search")
class WebSearchTool(BaseTool):
    """Search the web: SearXNG (preferred) → Tavily → DuckDuckGo."""

    tool_id = "web_search"
    is_local = False

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        searxng_url: str = "",
    ):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._max_results = max_results
        # SearXNG is opt-in by URL: unset means the tool behaves exactly
        # as it did before (Tavily → DDG), so nothing breaks for an
        # instance that never starts the container.
        self._searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description=(
                "Search the web for current information."
                " Returns relevant search results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return.",
                    },
                },
                "required": ["query"],
            },
            category="search",
            metadata={"requires_api_key": "TAVILY_API_KEY", "fallback": "duckduckgo"},
        )

    @staticmethod
    def _is_url(text: str) -> bool:
        """Check if text is a URL."""
        stripped = text.strip()
        return stripped.startswith("http://") or stripped.startswith("https://")

    @staticmethod
    def _extract_url(text: str) -> str | None:
        """Extract the first URL from text, if any."""
        import re as _re

        match = _re.search(r"https?://[^\s,;\"'<>]+", text)
        return match.group(0).rstrip(".,;)") if match else None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Convert known PDF URLs to their HTML equivalents."""
        import re as _re

        # arxiv: /pdf/ID → /abs/ID (abstract page with full metadata)
        m = _re.match(r"(https?://arxiv\.org)/pdf/(.+?)(?:\.pdf)?$", url)
        if m:
            return f"{m.group(1)}/abs/{m.group(2)}"
        return url

    @staticmethod
    def _fetch_url(url: str, max_chars: int = 6000) -> str:
        """Fetch a URL and return extracted text content."""
        import re as _re

        import httpx

        url = WebSearchTool._normalize_url(url)
        ssrf_error = check_ssrf(url)
        if ssrf_error:
            raise ValueError(ssrf_error)
        resp = httpx.get(
            url.strip(),
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; OpenJarvis/1.0; +https://github.com/openjarvis)"
            },
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/pdf" in content_type:
            return (
                "[This URL points to a PDF file which"
                f" cannot be read directly. URL: {url}]"
            )
        html = resp.text
        # Strip script/style tags and their contents
        html = _re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            "",
            html,
            flags=_re.DOTALL | _re.IGNORECASE,
        )
        # Strip HTML tags
        text = _re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = _re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated]"
        return text

    def _searxng_search(self, query: str, max_results: int) -> tuple[str, int]:
        """Search via a self-hosted SearXNG instance (JSON API).

        Preferred backend (Brique 3, docs/SPEC_BRIQUE3_RECHERCHE.md):
        multi-source (Google/Bing/GitHub/StackOverflow/HN/arXiv/Wikipedia),
        free, self-hosted, and measured faster than DDG alone
        (0.45-0.85 s/req vs 2.16 s) for ~120 MB of RAM -- negligible next
        to this project's heavy pipelines.

        Integration trap worth knowing: SearXNG's *own* DuckDuckGo engine
        returns 0 results. That is exactly why DDG stays wired as a
        **direct** fallback (ddgs) rather than as a SearXNG engine.
        """
        import httpx

        resp = httpx.get(
            self._searxng_url.rstrip("/") + "/search",
            params={"q": query, "format": "json"},
            timeout=_SEARXNG_TIMEOUT_S,
        )
        resp.raise_for_status()
        results = (resp.json().get("results") or [])[:max_results]

        parts = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", "") or ""
            engines = ", ".join(r.get("engines") or []) or r.get("engine", "")
            suffix = f"\nEngines: {engines}" if engines else ""
            parts.append(f"### {title}\nSource: {url}\nSummary: {snippet}{suffix}")
        return "\n\n---\n\n".join(parts), len(results)

    def _duckduckgo_search(self, query: str, max_results: int) -> str:
        """Search using DuckDuckGo as a direct fallback.

        Locale is forced to French (Brique 3 decision): DDG otherwise
        defaults to English results, the wrong default for a user who
        talks to JARVIS in French. SearXNG needs no equivalent -- it is
        locale-agnostic by construction.
        """
        from ddgs import DDGS

        ddgs = DDGS()
        try:
            raw_results = list(
                ddgs.text(query, max_results=max_results, region=_DDG_REGION)
            )
        except TypeError:
            # ddgs versions differ on the region kwarg name; getting the
            # results at all matters more than the locale hint.
            raw_results = list(ddgs.text(query, max_results=max_results))
        results = []
        for r in raw_results:
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            snippet = r.get("body", "")
            results.append(f"### {title}\nSource: {url}\nSummary: {snippet}")

        formatted = "\n\n---\n\n".join(results)
        return formatted

    def execute(self, **params: Any) -> ToolResult:
        query = params.get("query", "")
        if not query:
            return ToolResult(
                tool_name="web_search",
                content="No query provided.",
                success=False,
            )

        # If the query contains a URL, fetch it directly instead of searching
        url = self._extract_url(query) if not self._is_url(query) else query.strip()
        if url:
            try:
                content = self._fetch_url(url)
                return ToolResult(
                    tool_name="web_search",
                    content=content or "No content found at URL.",
                    success=True,
                    metadata={"url": url, "mode": "fetch"},
                )
            except Exception as exc:
                return ToolResult(
                    tool_name="web_search",
                    content=f"Failed to fetch URL: {exc}",
                    success=False,
                )

        max_results = params.get("max_results", self._max_results)

        # Preferred backend: self-hosted SearXNG (free, multi-source,
        # faster than DDG alone). Falls through to Tavily/DDG below on any
        # failure, so a stopped container degrades instead of breaking.
        if self._searxng_url:
            try:
                formatted, count = self._searxng_search(query, max_results)
                if count:
                    return ToolResult(
                        tool_name="web_search",
                        content=formatted,
                        success=True,
                        metadata={"engine": "searxng", "num_results": count},
                    )
                logger.debug("SearXNG returned no results, falling through")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "SearXNG error (%s), falling back", type(exc).__name__
                )

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._api_key)
            response = client.search(
                query,
                max_results=max_results,
                search_depth="advanced",
                include_usage=True,
            )
            results = response.get("results", [])
            formatted_parts = []
            for r in results:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = r.get("content", "") or r.get("snippet", "")
                formatted_parts.append(
                    f"### {title}\nSource: {url}\nSummary: {content}"
                )

            formatted = "\n\n---\n\n".join(formatted_parts)
            return ToolResult(
                tool_name="web_search",
                content=formatted or "No results found.",
                success=True,
                metadata={
                    "num_results": len(results),
                    "engine": "tavily",
                    "credits": (response.get("usage") or {}).get("credits"),
                },
            )
        except Exception as exc:
            logger.debug(
                "Tavily error (%s), falling back to DuckDuckGo", type(exc).__name__
            )

        try:
            formatted = self._duckduckgo_search(query, max_results)
            return ToolResult(
                tool_name="web_search",
                content=formatted or "No results found.",
                success=True,
                metadata={"engine": "duckduckgo"},
            )
        except ImportError:
            return ToolResult(
                tool_name="web_search",
                content=(
                    "tavily-python not installed and ddgs not available."
                    " Install with: pip install tavily-python ddgs"
                ),
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name="web_search",
                content=f"Search error: {exc}",
                success=False,
            )


__all__ = ["WebSearchTool"]
