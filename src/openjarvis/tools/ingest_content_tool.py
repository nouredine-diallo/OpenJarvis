"""Agent-facing tool wrapping the universal ingestion router (Brique 1).

Exposes ``ingest_content`` (tools/ingest_router.py) as a callable tool so
the orchestrator can hand it anything a user references -- a GitHub repo
URL, a PDF/image/audio file path, a YouTube link, a plain web URL -- and
get back normalized Markdown with sourced metadata, without the user
having to say what to do with it (spec §1: "sans que l'utilisateur
explique quoi lire").
"""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.ingest_router import ingest_content


@ToolRegistry.register("ingest_content")
class IngestContentTool(BaseTool):
    """Detect and ingest a repo/PDF/image/audio/video/web/text source."""

    tool_id = "ingest_content"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ingest_content",
            description=(
                "Ingest any content the user references -- a GitHub repo "
                "URL or local repo folder, a PDF, an image (OCR), an audio "
                "file, a YouTube link, a plain web URL, or a local text/"
                "markdown file -- and return it as normalized Markdown with "
                "sourced metadata. This tool runs directly on the same "
                "machine as the rest of your tools (same as file_read, "
                "shell_exec): it CAN read local file paths the user gives "
                "you, there is no 'no filesystem access' limitation here. "
                "Use this whenever the user gives you something to look at "
                "('analyse ce repo', 'lis ce PDF', 'regarde cette image', "
                "'voilà une vidéo', or a bare local path/URL) -- call it "
                "immediately with that path/URL as `source` instead of "
                "asking the user to paste the content or declining because "
                "you assume you can't reach it. A video with no subtitles "
                "or transcription is returned as metadata-only, explicitly "
                "marked as not analyzed -- never invented."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "A URL (GitHub repo, YouTube, web page) or a "
                            "local file/folder path."
                        ),
                    },
                },
                "required": ["source"],
            },
            category="media",
        )

    def execute(self, **params: Any) -> ToolResult:
        source = str(params.get("source", "") or "").strip()
        if not source:
            return ToolResult(tool_name=self.tool_id, content="No source provided.", success=False)

        result = ingest_content(source)
        if not result.success:
            return ToolResult(tool_name=self.tool_id, content=result.error, success=False)

        content = f"{result.to_markdown_header()}\n\n{result.content}"
        return ToolResult(
            tool_name=self.tool_id,
            content=content,
            success=True,
            metadata={
                "content_type": result.content_type,
                "mode": result.mode,
                "source": result.source,
                **result.metadata,
            },
        )


__all__ = ["IngestContentTool"]
