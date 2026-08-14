"""Tests for the ingest_content agent tool (thin wrapper around
tools/ingest_router.py -- pipeline correctness is tested there)."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools.ingest_content_tool import IngestContentTool


def _make_tool() -> IngestContentTool:
    if not ToolRegistry.contains("ingest_content"):
        ToolRegistry.register_value("ingest_content", IngestContentTool)
    return IngestContentTool()


def test_registered_in_tool_registry():
    ToolRegistry.register_value("ingest_content", IngestContentTool)
    assert ToolRegistry.contains("ingest_content")


def test_missing_source_fails_cleanly():
    tool = _make_tool()
    result = tool.execute()
    assert result.success is False


def test_real_text_file_ingestion_includes_metadata_header(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("Never deploy on a Friday.")
    tool = _make_tool()
    result = tool.execute(source=str(f))
    assert result.success is True
    assert "Never deploy" in result.content
    assert "source:" in result.content  # metadata header present
    assert result.metadata["content_type"] == "text"


def test_unresolvable_source_fails_with_error_message():
    tool = _make_tool()
    result = tool.execute(source="totally unrecognizable input")
    assert result.success is False
    assert result.content  # human-readable error, not empty
