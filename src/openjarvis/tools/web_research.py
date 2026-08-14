"""Multi-hop web research with a cited synthesis.

Brique 3 (docs/SPEC_BRIQUE3_RECHERCHE.md). The engine already had two
research agents, but both search the *personal corpus* only
(``deep_research``, ``research_loop``); the only web path was a single
``web_search`` call returning a flat list of links. The spec's own
judgement: "liste de liens = insuffisant ; synthèse citée = la vraie
valeur". This closes that gap.

The loop, deliberately small and budget-bounded:

    question -> plan sub-queries -> search (SearXNG, multi-source)
             -> fetch the most promising pages -> synthesize WITH citations

Anti-hallucination rules, matching how the rest of this project treats
evidence (PROJET_JARVIS.md §3, missions/verifier.py):

* the synthesis prompt is given the retrieved extracts and instructed to
  use *only* those -- not the model's own recollection;
* every claim must carry a ``[n]`` citation, and the numbered sources are
  returned alongside so any claim is traceable to a URL;
* when nothing is retrieved, it says so instead of answering from memory.

Budget matters as much as quality here: each mission has a 50k token
ceiling, so sub-queries, fetched pages and extract length are all capped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

_MAX_SUBQUERIES = 3
_RESULTS_PER_QUERY = 5
_MAX_PAGES_FETCHED = 3
_EXTRACT_CHARS = 1500
# qwen3.6 reasons inline and can burn a small budget entirely on <think>,
# leaving no answer -- give the synthesis enough room to finish thinking
# *and* write.
_SYNTHESIS_MAX_TOKENS = 4096

_PLAN_PROMPT = (
    "Tu prépares une recherche web. Pour la question de l'utilisateur, "
    "propose entre 1 et {n} requêtes de recherche courtes et "
    "complémentaires (angles différents, pas des reformulations). "
    "Réponds UNIQUEMENT avec un tableau JSON de chaînes."
)

_SYNTHESIS_PROMPT = (
    "Tu rédiges une synthèse de recherche à partir d'extraits web.\n"
    "RÈGLES STRICTES :\n"
    "- Utilise UNIQUEMENT les extraits fournis. N'ajoute aucune "
    "connaissance personnelle.\n"
    "- Chaque affirmation doit être suivie de sa source au format [n].\n"
    "- Si les extraits ne permettent pas de répondre, dis-le explicitement "
    "au lieu d'inventer.\n"
    "- Sois concis et factuel. Réponds en français.\n"
    "- Termine par les limites de cette recherche (ce qui reste incertain)."
)


@dataclass(slots=True)
class Source:
    """A numbered, citable source."""

    ref: int
    title: str
    url: str
    snippet: str = ""

    def as_line(self) -> str:
        return f"[{self.ref}] {self.title} — {self.url}"


@dataclass(slots=True)
class ResearchOutcome:
    answer: str = ""
    sources: List[Source] = field(default_factory=list)
    queries_run: List[str] = field(default_factory=list)
    pages_fetched: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and bool(self.answer)


def _strip_reasoning(content: str) -> str:
    """Remove the model's inline chain of thought.

    Handles the *unclosed* case as well as the normal one: qwen3.6-27b can
    spend its whole token budget reasoning and get truncated before it ever
    emits </think>. A closing-tag-only regex leaves that entire block in
    place, which is how raw reasoning ended up presented as a research
    synthesis (observed live). When the block never closes there is no
    answer at all -- returning empty is correct, and the caller then
    reports failure instead of passing reasoning off as a result.
    """
    content = content or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<think>.*\Z", "", content, flags=re.DOTALL | re.IGNORECASE)
    return content


def _parse_json_list(content: str) -> List[str]:
    """Extract a JSON array of strings from model output.

    Strips <think> blocks first: the default model (qwen3.6-27b) emits its
    chain of thought inline, and letting it through here would turn
    reasoning lines into search queries -- the same class of bug that
    silently filled the fact store with junk (memory/extractor.py).
    """
    import json

    content = _strip_reasoning(content)
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _parse_search_results(formatted: str) -> List[Dict[str, str]]:
    """Parse WebSearchTool's markdown output back into structured hits.

    The tool returns display-formatted text (``### title`` / ``Source:``
    / ``Summary:``) rather than structured data; parsing it here keeps
    this module from having to duplicate the multi-backend search logic.
    """
    hits: List[Dict[str, str]] = []
    for block in (formatted or "").split("\n\n---\n\n"):
        title = url = snippet = ""
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("### "):
                title = line[4:].strip()
            elif line.startswith("Source: "):
                url = line[8:].strip()
            elif line.startswith("Summary: "):
                snippet = line[9:].strip()
        if url:
            hits.append({"title": title or url, "url": url, "snippet": snippet})
    return hits


class WebResearcher:
    """Runs the plan -> search -> fetch -> cite loop."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        search_tool: Any = None,
        max_subqueries: int = _MAX_SUBQUERIES,
        max_pages: int = _MAX_PAGES_FETCHED,
    ) -> None:
        self._engine = engine
        self._model = model
        self._search_tool = search_tool
        self._max_subqueries = max_subqueries
        self._max_pages = max_pages

    def _search_tool_or_default(self) -> Any:
        if self._search_tool is None:
            from openjarvis.tools.web_search import WebSearchTool

            self._search_tool = WebSearchTool()
        return self._search_tool

    def _ask(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        result = self._engine.generate(
            [Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=user)],
            model=self._model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        return _strip_reasoning(content).strip()

    def plan(self, question: str) -> List[str]:
        """Break the question into complementary sub-queries.

        Falls back to the raw question when planning fails -- a degraded
        single-query search is far better than no research at all.
        """
        try:
            raw = self._ask(_PLAN_PROMPT.format(n=self._max_subqueries), question, max_tokens=256)
            queries = _parse_json_list(raw)[: self._max_subqueries]
        except Exception:  # noqa: BLE001
            logger.debug("Sub-query planning failed", exc_info=True)
            queries = []
        return queries or [question]

    def research(self, question: str) -> ResearchOutcome:
        outcome = ResearchOutcome()
        tool = self._search_tool_or_default()

        seen_urls: set[str] = set()
        sources: List[Source] = []
        for query in self.plan(question):
            outcome.queries_run.append(query)
            try:
                res = tool.execute(query=query, max_results=_RESULTS_PER_QUERY)
            except Exception:  # noqa: BLE001
                logger.debug("Search failed for %r", query, exc_info=True)
                continue
            if not res.success:
                continue
            for hit in _parse_search_results(res.content):
                if hit["url"] in seen_urls:
                    continue
                seen_urls.add(hit["url"])
                sources.append(
                    Source(ref=len(sources) + 1, title=hit["title"], url=hit["url"], snippet=hit["snippet"])
                )

        if not sources:
            outcome.error = "Aucun résultat de recherche exploitable."
            return outcome

        # Attach sources as soon as they exist, not only on the success
        # path: when synthesis later fails, the caller should still learn
        # what was actually retrieved instead of seeing "0 sources" and
        # concluding the search itself found nothing.
        outcome.sources = sources

        # Fetch the top few pages for real content rather than relying on
        # snippets alone -- snippets are often truncated mid-sentence.
        extracts: List[str] = []
        for src in sources[: self._max_pages]:
            body = self._fetch(tool, src.url)
            if body:
                outcome.pages_fetched += 1
                extracts.append(f"[{src.ref}] {src.title}\n{body[:_EXTRACT_CHARS]}")
            else:
                extracts.append(f"[{src.ref}] {src.title}\n{src.snippet}")
        for src in sources[self._max_pages :]:
            if src.snippet:
                extracts.append(f"[{src.ref}] {src.title}\n{src.snippet}")

        payload = f"QUESTION : {question}\n\nEXTRAITS :\n\n" + "\n\n---\n\n".join(extracts)
        try:
            outcome.answer = self._ask(_SYNTHESIS_PROMPT, payload, max_tokens=_SYNTHESIS_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"Synthèse impossible : {exc}"
            return outcome

        if not outcome.answer:
            # Everything the model produced was reasoning that got cut off
            # mid-thought. Report that honestly rather than returning an
            # empty "synthesis" that looks like a real answer.
            outcome.error = (
                "Le modèle n'a pas produit de synthèse exploitable "
                "(raisonnement tronqué). Sources trouvées ci-dessous."
            )
            return outcome

        return outcome

    @staticmethod
    def _fetch(tool: Any, url: str) -> str:
        """Fetch a page through the search tool's own URL mode, which
        already carries the SSRF guard (spec §4.3 point 4)."""
        try:
            res = tool.execute(query=url)
            return res.content if res.success else ""
        except Exception:  # noqa: BLE001
            logger.debug("Page fetch failed for %s", url, exc_info=True)
            return ""


@ToolRegistry.register("deep_web_research")
class DeepWebResearchTool(BaseTool):
    """Multi-hop web research returning a cited synthesis."""

    tool_id = "deep_web_research"
    is_local = False
    #: Injected post-build in cli/serve.py, same pattern as the mission tools.
    _engine: Optional[Any] = None
    _model: str = ""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="deep_web_research",
            description=(
                "Research a question on the web across several sources and "
                "return a synthesis where every claim is cited [1][2] with "
                "a numbered source list. Use this for open or comparative "
                "questions ('quelles solutions existent pour X', 'compare A "
                "et B', 'quel est l'état de l'art de Y') where a plain "
                "web_search list of links would not be enough. Slower than "
                "web_search (several searches plus page fetches), so prefer "
                "web_search for a single quick lookup."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question, in natural language.",
                    },
                },
                "required": ["question"],
            },
            category="search",
        )

    def execute(self, **params: Any) -> ToolResult:
        question = str(params.get("question", "") or "").strip()
        if not question:
            return ToolResult(tool_name=self.tool_id, content="No question provided.", success=False)
        if self._engine is None or not self._model:
            return ToolResult(
                tool_name=self.tool_id,
                content="Moteur LLM non disponible pour la recherche approfondie.",
                success=False,
            )

        outcome = WebResearcher(self._engine, self._model).research(question)
        if not outcome.success:
            return ToolResult(
                tool_name=self.tool_id,
                content=outcome.error or "Recherche sans résultat.",
                success=False,
                metadata={"queries_run": outcome.queries_run},
            )

        body = outcome.answer + "\n\n## Sources\n" + "\n".join(s.as_line() for s in outcome.sources)
        return ToolResult(
            tool_name=self.tool_id,
            content=body,
            success=True,
            metadata={
                "queries_run": outcome.queries_run,
                "num_sources": len(outcome.sources),
                "pages_fetched": outcome.pages_fetched,
            },
        )


__all__ = ["DeepWebResearchTool", "ResearchOutcome", "Source", "WebResearcher"]
