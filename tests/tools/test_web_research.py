"""Tests for multi-hop web research with cited synthesis (Brique 3).

The LLM and the search backend are faked here on purpose: what matters is
the *loop's* contract -- budget caps, deduplication, citation numbering,
and above all that it degrades honestly (says "nothing found") instead of
answering from the model's own memory when retrieval turns up empty.
Real-backend behaviour (SearXNG, DDG fallback) is covered in
test_web_search.py.
"""

from __future__ import annotations

from openjarvis.core.types import ToolResult
from openjarvis.tools.web_research import (
    DeepWebResearchTool,
    Source,
    WebResearcher,
    _parse_json_list,
    _parse_search_results,
)


class FakeEngine:
    """Returns queued responses in order; records the prompts it saw."""

    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, messages, **kwargs):
        self.prompts.append("\n".join(m.content for m in messages))
        content = self._responses.pop(0) if self._responses else ""
        return {"content": content}


class FakeSearchTool:
    def __init__(self, results_by_query=None, fetch_body="", fail=False):
        self._results = results_by_query or {}
        self._fetch_body = fetch_body
        self._fail = fail
        self.calls = []

    def execute(self, **params):
        query = params.get("query", "")
        self.calls.append(query)
        if self._fail:
            return ToolResult(tool_name="web_search", content="boom", success=False)
        if query.startswith("http"):
            return ToolResult(
                tool_name="web_search", content=self._fetch_body, success=bool(self._fetch_body)
            )
        content = self._results.get(query, "")
        max_results = params.get("max_results")
        if max_results and content:
            blocks = content.split("\n\n---\n\n")[:max_results]
            content = "\n\n---\n\n".join(blocks)
        return ToolResult(tool_name="web_search", content=content, success=True)


def _block(title, url, summary):
    return f"### {title}\nSource: {url}\nSummary: {summary}"


class TestParsing:
    def test_parse_search_results_extracts_structured_hits(self):
        formatted = "\n\n---\n\n".join(
            [_block("Leon", "https://getleon.ai/", "Open-source assistant"),
             _block("Khoj", "https://khoj.dev/", "Second brain")]
        )
        hits = _parse_search_results(formatted)
        assert [h["url"] for h in hits] == ["https://getleon.ai/", "https://khoj.dev/"]
        assert hits[0]["title"] == "Leon"

    def test_parse_search_results_ignores_blocks_without_url(self):
        assert _parse_search_results("### Title only\nSummary: no source") == []

    def test_parse_json_list_strips_think_blocks(self):
        """qwen3.6 emits <think>...</think> inline; letting it through would
        turn reasoning lines into search queries."""
        raw = '<think>Let me plan. What angles?</think>["query one", "query two"]'
        assert _parse_json_list(raw) == ["query one", "query two"]

    def test_parse_json_list_returns_empty_on_prose(self):
        assert _parse_json_list("I think we should search for stuff.") == []


class TestPlanning:
    def test_plan_uses_model_subqueries(self):
        engine = FakeEngine('["angle A", "angle B"]')
        assert WebResearcher(engine, "m").plan("question") == ["angle A", "angle B"]

    def test_plan_caps_subqueries_to_budget(self):
        engine = FakeEngine('["a", "b", "c", "d", "e"]')
        assert len(WebResearcher(engine, "m", max_subqueries=2).plan("q")) == 2

    def test_plan_falls_back_to_raw_question(self):
        """A degraded single-query search beats no research at all."""
        engine = FakeEngine("not json at all")
        assert WebResearcher(engine, "m").plan("ma question") == ["ma question"]

    def test_plan_survives_engine_failure(self):
        class Boom:
            def generate(self, *a, **k):
                raise RuntimeError("engine down")

        assert WebResearcher(Boom(), "m").plan("ma question") == ["ma question"]


class TestResearchLoop:
    def test_produces_cited_synthesis_with_numbered_sources(self):
        engine = FakeEngine('["assistants open source"]', "Leon est self-hosted [1].")
        search = FakeSearchTool(
            {"assistants open source": _block("Leon", "https://getleon.ai/", "Open-source")},
            fetch_body="Leon runs on your own server.",
        )
        outcome = WebResearcher(engine, "m", search_tool=search).research("quelles solutions ?")

        assert outcome.success is True
        assert "[1]" in outcome.answer
        assert [s.ref for s in outcome.sources] == [1]
        assert outcome.sources[0].url == "https://getleon.ai/"

    def test_deduplicates_urls_across_subqueries(self):
        dup = _block("Leon", "https://getleon.ai/", "Open-source")
        engine = FakeEngine('["q1", "q2"]', "synthèse [1]")
        search = FakeSearchTool({"q1": dup, "q2": dup}, fetch_body="body")
        outcome = WebResearcher(engine, "m", search_tool=search).research("q")
        assert len(outcome.sources) == 1

    def test_citation_refs_are_sequential_across_queries(self):
        engine = FakeEngine('["q1", "q2"]', "a [1] b [2]")
        search = FakeSearchTool(
            {
                "q1": _block("A", "https://a.example/", "sa"),
                "q2": _block("B", "https://b.example/", "sb"),
            },
            fetch_body="body",
        )
        outcome = WebResearcher(engine, "m", search_tool=search).research("q")
        assert [s.ref for s in outcome.sources] == [1, 2]

    def test_no_results_reports_failure_instead_of_answering(self):
        """The anti-hallucination contract: with nothing retrieved it must
        refuse, never fall back on the model's own knowledge."""
        engine = FakeEngine('["q"]', "I happen to know the answer anyway.")
        search = FakeSearchTool({"q": ""})
        outcome = WebResearcher(engine, "m", search_tool=search).research("q")
        assert outcome.success is False
        assert outcome.error
        assert outcome.answer == ""

    def test_search_failure_is_survived(self):
        engine = FakeEngine('["q"]', "answer")
        outcome = WebResearcher(engine, "m", search_tool=FakeSearchTool(fail=True)).research("q")
        assert outcome.success is False

    def test_falls_back_to_snippet_when_page_fetch_fails(self):
        engine = FakeEngine('["q"]', "synthèse [1]")
        search = FakeSearchTool(
            {"q": _block("A", "https://a.example/", "le snippet utile")}, fetch_body=""
        )
        outcome = WebResearcher(engine, "m", search_tool=search).research("q")
        assert outcome.success is True
        assert outcome.pages_fetched == 0
        assert "le snippet utile" in engine.prompts[-1]

    def test_page_fetch_count_is_capped(self):
        blocks = "\n\n---\n\n".join(
            _block(f"T{i}", f"https://e{i}.example/", "s") for i in range(6)
        )
        engine = FakeEngine('["q"]', "synthèse")
        search = FakeSearchTool({"q": blocks}, fetch_body="body")
        outcome = WebResearcher(engine, "m", search_tool=search, max_pages=2).research("q")
        # Only max_pages pages are actually fetched, but every hit still
        # becomes a citable source (from its snippet) -- the fetch budget
        # limits cost, it must not silently drop sources.
        assert outcome.pages_fetched == 2
        assert len(outcome.sources) == 5  # search tool's own max_results cap

    def test_synthesis_prompt_forbids_outside_knowledge(self):
        engine = FakeEngine('["q"]', "synthèse [1]")
        search = FakeSearchTool({"q": _block("A", "https://a.example/", "s")}, fetch_body="b")
        WebResearcher(engine, "m", search_tool=search).research("q")
        assert "UNIQUEMENT les extraits" in engine.prompts[-1]


class TestTool:
    def test_missing_question_fails(self):
        assert DeepWebResearchTool().execute().success is False

    def test_missing_engine_fails_cleanly(self):
        tool = DeepWebResearchTool()
        tool._engine = None
        result = tool.execute(question="q")
        assert result.success is False
        assert "Moteur" in result.content

    def test_successful_result_appends_source_list(self, monkeypatch):
        """Patches WebResearcher in the module namespace (what execute()
        actually resolves) via the monkeypatch fixture -- a hand-rolled
        class-attribute swap proved unreliable under the full suite and let
        this unit test fire a real network request."""
        from openjarvis.tools import web_research
        from openjarvis.tools.web_research import ResearchOutcome

        class FakeResearcher:
            def __init__(self, *a, **k):
                pass

            def research(self, question):
                return ResearchOutcome(
                    answer="synthèse [1]",
                    sources=[Source(ref=1, title="A", url="https://a.example/")],
                    queries_run=["q"],
                )

        monkeypatch.setattr(web_research, "WebResearcher", FakeResearcher)

        tool = DeepWebResearchTool()
        tool._engine = FakeEngine()
        tool._model = "m"
        result = tool.execute(question="q")

        assert result.success is True
        assert "## Sources" in result.content
        assert "https://a.example/" in result.content
        assert result.metadata["num_sources"] == 1

    def test_sources_are_reported_even_when_synthesis_fails(self):
        """A synthesis failure must not make it look like the search found
        nothing -- the retrieved sources are still useful information."""

        class HalfBrokenEngine:
            def __init__(self):
                self.calls = 0

            def generate(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {"content": '["q"]'}
                raise RuntimeError("LLM unavailable")

        search = FakeSearchTool(
            {"q": _block("A", "https://a.example/", "s")}, fetch_body="body"
        )
        outcome = WebResearcher(HalfBrokenEngine(), "m", search_tool=search).research("q")
        assert outcome.success is False
        assert "Synthèse impossible" in outcome.error
        assert len(outcome.sources) == 1

    def test_truncated_reasoning_is_reported_not_passed_off_as_answer(self):
        """Observed live: qwen3.6 spent its whole budget reasoning and was
        cut off before </think>, so raw chain-of-thought was returned as if
        it were the research synthesis. It must fail honestly instead --
        while still surfacing the sources it did find."""
        engine = FakeEngine('["q"]', "<think>Here's a thinking process: 1. Analyze")
        search = FakeSearchTool(
            {"q": _block("A", "https://a.example/", "s")}, fetch_body="body"
        )
        outcome = WebResearcher(engine, "m", search_tool=search).research("q")
        assert outcome.success is False
        assert "tronqué" in outcome.error
        assert "<think>" not in outcome.answer
        assert len(outcome.sources) == 1
