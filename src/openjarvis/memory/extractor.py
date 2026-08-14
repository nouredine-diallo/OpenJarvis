"""LLM-backed extraction of durable facts from a conversation turn.

The extractor takes a single (user, assistant) exchange and asks a small
local model to distill any long-term, user-specific facts worth remembering.
It is deliberately defensive: extraction runs on a background thread far from
the request path, so *any* failure — a dropped Ollama connection, a timeout, a
``BrokenPipeError`` when the client went away, or simply unparseable output —
must degrade to "no facts" rather than propagate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from openjarvis.core.types import Message, Role
from openjarvis.memory.store import (
    Fact,
    KIND_DECISION,
    KIND_FACT,
    KIND_PREFERENCE,
    KIND_RULE,
    normalize_kind,
)

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You extract durable, long-term facts about the user from a single "
    "conversation exchange. A good fact is stable over time and useful in "
    "future conversations: preferences, identity, goals, ongoing projects, "
    "constraints, or relationships. Ignore one-off task details, small talk, "
    "and anything the assistant said about itself.\n\n"
    "Respond with ONLY a JSON array of short fact strings (each under 200 "
    "characters). If there is nothing worth remembering, respond with []."
)

# Typed extraction (spec §6/§8): each memory is classified as a preference,
# fact, decision or rule so the 4 stores stay distinct.
_DEFAULT_TYPED_SYSTEM_PROMPT = (
    "You maintain the user's long-term memory. From a single conversation "
    "exchange, extract durable statements worth remembering and classify each "
    "into EXACTLY one of these kinds:\n"
    f"- \"{KIND_PREFERENCE}\": a stable user preference or style choice "
    "(e.g. \"Prefers concise French answers\").\n"
    f"- \"{KIND_FACT}\": a durable, dated, factual statement about the user "
    "or their world (identity, projects, relationships, devices).\n"
    f"- \"{KIND_DECISION}\": a decision made and locked (e.g. \"Chose "
    "OpenJarvis over jarvis-OS as the base\").\n"
    f"- \"{KIND_RULE}\": an explicit, executable convention the assistant "
    "must follow (e.g. \"Never declare a task done without running tests\").\n"
    "Ignore one-off task details and small talk.\n\n"
    "Respond with ONLY a JSON array of objects: "
    '[{"kind": "preference", "text": "..."}] — text under 200 characters. '
    'If there is nothing worth remembering, respond with [].'
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: Openers that mark a line as the model *narrating the exchange* rather
#: than stating a durable fact about the user. Collected from real
#: polluted entries found in the live fact store on 2026-08-14.
_NARRATION_PREFIXES = (
    "user says",
    "user asks",
    "user mentions",
    "the user says",
    "the user asks",
    "the user mentions",
    "assistant:",
    "the assistant",
    "assistant explains",
    "assistant responds",
    "this is a test",
    "this response",
    "this exchange",
    "the prompt",
    "no explicit",
    "nothing worth",
    "there is nothing",
    "no durable",
    "no preferences",
    "it asks",
    "offers to",
    "notes that",
    "here's a",
    "here is a",
    "okay,",
    "let me",
    "i need to",
    "wait,",
    # Descriptions *of the exchange* rather than facts about the user's
    # world. Kept as full phrases on purpose: a bare "the user's" prefix
    # would also reject legitimate entries like "The user's laptop has
    # 7.6 GB of RAM".
    "the conversation",
    "the user said",
    "the user's input",
    "this is a factual statement",
    "user provides",
    "the url is",
)


def _strip_reasoning(content: str) -> str:
    """Remove <think>...</think> blocks.

    The default model (qwen3.6-27b) emits its chain of thought inline in
    exactly this form. Left in place, that reasoning reaches the
    line-based fallback below and every sentence of it gets stored as a
    "durable fact" -- which is precisely how the live fact store ended up
    full of "Does it contain a stable preference? No.". The same fix was
    already needed in the Cloudflare control plane's /ask endpoint; this
    is the same model behaviour biting a second code path.
    """
    return _THINK_RE.sub("", content or "")


def _looks_like_narration(text: str) -> bool:
    """True when *text* reads as commentary about the conversation rather
    than a durable fact worth remembering.

    Deliberately conservative in what it *rejects* on top of the prefix
    list: only questions (a fact is never a question) and the model's
    explicit "nothing to store" answers, both of which are worthless as
    memory by construction and were observed live.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith(_NARRATION_PREFIXES):
        return True
    # A durable fact is a declarative statement, so it contains no question
    # mark anywhere -- not just at the end. Checking only the end let the
    # model's self-Q&A reasoning through verbatim ("Does it contain a stable
    # preference? No."), which is exactly the shape that polluted the live
    # store. Facts that legitimately embed a question are vanishingly rare
    # next to reasoning text, which is full of them.
    if "?" in stripped:
        return True
    # Leftover reasoning markup: a durable fact never carries think tags.
    if "<think>" in lowered or "</think>" in lowered:
        return True
    # A label/heading ("**Analyze User Input:**", "Step 2:") introduces
    # content rather than being content -- real entries found in the live
    # store came from the model's own markdown reasoning headers.
    if stripped.rstrip("*").endswith(":"):
        return True
    return False


class FactExtractor:
    """Extract memory-worthy facts from a conversation turn via an engine."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_facts_per_turn: int = 10,
        max_fact_chars: int = 200,
        system_prompt: Optional[str] = None,
        typed_system_prompt: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_facts_per_turn = max_facts_per_turn
        self._max_fact_chars = max_fact_chars
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._typed_system_prompt = typed_system_prompt or _DEFAULT_TYPED_SYSTEM_PROMPT

    def extract(self, user_text: str, assistant_text: str = "") -> List[str]:
        """Return durable facts (plain strings) from the exchange.

        Delegates to :meth:`extract_typed`; entries classified as facts are
        returned as their text. Never raises.
        """
        return [f.text for f in self.extract_typed(user_text, assistant_text)]

    def extract_typed(self, user_text: str, assistant_text: str = "") -> List[Fact]:
        """Return typed memory entries (kind + text). Never raises."""
        user_text = (user_text or "").strip()
        if not user_text:
            return []

        exchange = f"User: {user_text}"
        if assistant_text and assistant_text.strip():
            exchange += f"\nAssistant: {assistant_text.strip()}"

        messages = [
            Message(role=Role.SYSTEM, content=self._typed_system_prompt),
            Message(role=Role.USER, content=exchange),
        ]

        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except BrokenPipeError:
            logger.debug("Memory extraction aborted: broken pipe", exc_info=True)
            return []
        except Exception:  # noqa: BLE001 — extraction must never crash the worker
            logger.debug("Memory extraction failed", exc_info=True)
            return []

        if isinstance(result, dict):
            content = result.get("content", "") or ""
        else:
            content = str(result)

        return self._parse_typed(content)

    # -- parsing ------------------------------------------------------------

    def _parse_typed(self, content: str) -> List[Fact]:
        """Parse model output into typed, deduped, capped memory entries."""
        if not content or not content.strip():
            return []

        raw = self._coerce_to_list(content)

        facts: List[Fact] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                kind = normalize_kind(item.get("kind"))
                text = item.get("text", "")
            else:
                kind = KIND_FACT
                text = str(item)
            fact_text = self._clean_fact(text)
            if not fact_text:
                continue
            key = fact_text.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(Fact(text=fact_text, kind=kind))
            if len(facts) >= self._max_facts_per_turn:
                break
        return facts

    def _parse(self, content: str) -> List[str]:
        """Parse model output into a clean, deduped, capped list of facts."""
        if not content or not content.strip():
            return []

        raw = self._coerce_to_list(content)

        facts: List[str] = []
        seen: set[str] = set()
        for item in raw:
            fact = self._clean_fact(item)
            if not fact:
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= self._max_facts_per_turn:
                break
        return facts

    def _coerce_to_list(self, content: str) -> List[str]:
        """Best-effort conversion of model output to a list of strings."""
        content = _strip_reasoning(content)

        # 1. Try to locate and parse a JSON array anywhere in the output
        #    (models often wrap it in prose or code fences).
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    # Preserve dict items (typed output) — coerce scalars to str.
                    return [x if isinstance(x, dict) else str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Fall back to line-based parsing (markdown bullets / numbered).
        #    Deliberately conservative: see _looks_like_narration. Found
        #    live (2026-08-14) polluting the real fact store -- when the
        #    model answers in prose instead of JSON, every line of its
        #    reasoning was being stored as a "durable fact" ("User says:
        #    ...", "Does it contain a stable preference? No.").
        items: List[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            if _looks_like_narration(line):
                continue
            items.append(line)
        return items

    def _clean_fact(self, item: str) -> str:
        fact = str(item).strip().strip("\"'").strip()
        # Drop obvious non-facts the model sometimes emits.
        if not fact or fact.lower() in ("[]", "none", "n/a", "null"):
            return ""
        if _looks_like_narration(fact):
            return ""
        if len(fact) > self._max_fact_chars:
            fact = fact[: self._max_fact_chars].rstrip()
        return fact


__all__ = ["FactExtractor"]
