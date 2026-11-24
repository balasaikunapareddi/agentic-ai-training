"""LLM access for the training labs.

One thin wrapper around LiteLLM so every lab talks to the model the same way:
the model comes from the ``LLM_MODEL`` env var (default
``gemini/gemini-3.5-flash-lite``) and the key from ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``). A transient 503/429 from the free tier is retried once
after 20 seconds.

Stub mode (the default) never touches the network: :class:`StubLLM` answers
from the same JSON tool-call contract the live prompt asks for, so a lab
behaves identically offline and live — only the reasoning is real in live mode.
"""

from __future__ import annotations

import json
import os
import re
import time

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"


def model_name() -> str:
    """Model id from env, e.g. ``gemini/gemini-3.5-flash-lite``."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def chat(system: str, user: str, model: str | None = None) -> str:
    """One chat completion via LiteLLM. Retries a transient failure once."""
    import litellm

    chosen = model or model_name()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = litellm.completion(
                model=chosen, messages=messages, api_key=api_key(), timeout=60
            )
            return str(response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - retry transient provider errors
            last_error = exc
            text = str(exc)
            if attempt == 0 and ("503" in text or "429" in text or "overloaded" in text.lower()):
                time.sleep(20)
                continue
            raise
    raise RuntimeError(f"LLM call failed: {last_error}")


# ---------------------------------------------------------------------------
# Stub LLM: deterministic, offline, same JSON contract as the live prompts.
# ---------------------------------------------------------------------------

_CAPITALS = {
    "france": "Paris",
    "germany": "Berlin",
    "japan": "Tokyo",
    "india": "New Delhi",
    "canada": "Ottawa",
}


class StubLLM:
    """Rule-based stand-in for the model.

    Understands the same JSON contract the labs' prompts describe:
    ``{"tool": ..., "args": {...}}`` to call a tool, or ``{"answer": ...}``
    to answer directly. It is deliberately small — it exists so the loop,
    the tools, and the tests are exercisable with no key, not to fake
    intelligence.
    """

    def __init__(self, role: str = "agent"):
        self.role = role

    def complete(self, system: str, user: str) -> str:
        text = user.strip()
        low = text.lower()

        # Follow-up turn: a tool result was handed back; answer with it.
        returned = re.search(r"returned:\s*(.+?)(?:\n|$)", text)
        if returned:
            return json.dumps({"answer": f"The answer is {returned.group(1).strip()}."})

        # Synthesis turn: the user message carries retrieved context. This
        # check comes before tool matching: context can contain digit
        # patterns (e.g. a phone number) that look like arithmetic.
        if "context:" in low or "sources:" in low:
            return self._synthesize(text)

        # Tool-call turn: the user message carries the question.
        question = self._extract_question(text)
        q = question.lower()

        arith = re.search(r"(-?\d+(?:\.\d+)?\s*(?:\+|-|\*|x|/)\s*-?\d+(?:\.\d+)?(?:\s*(?:\+|-|\*|x|/)\s*-?\d+(?:\.\d+)?)*)", q)
        if "how many words" in q:
            quoted = re.search(r"[\"']([^\"']+)[\"']", question)
            target = quoted.group(1) if quoted else question.split("in", 1)[-1].strip(" ?\"'")
            return json.dumps({"tool": "word_count", "args": {"text": target}})
        if arith and any(w in q for w in ("what is", "calculate", "compute", "evaluate")):
            expr = arith.group(1).replace("x", "*")
            return json.dumps({"tool": "calculator", "args": {"expression": expr}})
        if "capital of" in q:
            for country, capital in _CAPITALS.items():
                if country in q:
                    return json.dumps({"answer": f"The capital of {country.title()} is {capital}."})
            return json.dumps({"answer": "I don't know the capital of that country."})

        # Researcher / writer / reviewer roles in Day 2 labs.
        if "planner" in self.role:
            return self._plan(text)
        if "researcher" in self.role:
            return self._research(text)
        if "writer" in self.role:
            return self._write(text)
        if "reviewer" in self.role:
            return self._review(text)
        return json.dumps({"answer": "I can help with arithmetic, word counts, and questions about the clinic knowledge base."})

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _extract_question(text: str) -> str:
        for marker in ("Question:", "QUESTION:"):
            if marker in text:
                return text.split(marker, 1)[-1].strip()
        return text

    @staticmethod
    def _synthesize(text: str) -> str:
        """Answer from the retrieved context: pick the most overlapping sentence.

        Every chunk in the context is prefixed ``[doc-id] ``; that tag is
        carried onto the chosen sentence, so the answer always cites the
        document its sentence actually came from. (Previously the tag only
        survived when the winning sentence happened to be the first sentence
        of its chunk — e.g. "The clinic is closed on Sundays." came back
        with no citation at all.) The ``Sources: ...`` trailer the template
        appends is not evidence, so it is dropped before scoring.
        """
        from agentic_ai_training.retrieval import _tokens

        context = text.split("Context:", 1)[-1] if "Context:" in text else text
        context = context.split("Sources:", 1)[0]
        question = text.split("Question:", 1)[-1].split("Context:", 1)[0].strip() if "Question:" in text else ""
        query_words = _tokens(question)
        parts = re.split(r"\[([a-z0-9-]+)\]\s*", context)
        segments: list[tuple[str, str]] = []
        if parts[0].strip():
            segments.append(("", parts[0]))
        for i in range(1, len(parts) - 1, 2):
            segments.append((parts[i], parts[i + 1]))
        best, best_score, best_tag = "", -1, ""
        for tag, chunk in segments:
            for sentence in re.split(r"(?<=[.!])\s+", chunk):
                sentence = sentence.strip()
                if not sentence:
                    continue
                score = len(query_words & _tokens(sentence))
                if score > best_score:
                    best, best_score, best_tag = sentence, score, tag
        if best_score <= 0:
            return "I couldn't find that in the clinic knowledge base."
        if best_tag and f"[{best_tag}]" not in best:
            return f"{best} [{best_tag}]"
        return best

    @staticmethod
    def _plan(text: str) -> str:
        return (
            "1. Current hours and access\n"
            "2. Telehealth today and the Saturday gap\n"
            "3. Recommendation"
        )

    @staticmethod
    def _research(text: str) -> str:
        return (
            "FACT: The clinic is open Monday-Friday 8:00 AM-6:00 PM and Saturday 9:00 AM-1:00 PM. [clinic-hours]\n"
            "FACT: Telehealth visits run Monday-Friday 9:00 AM-5:00 PM for follow-ups, medication refills, and minor illness. [telehealth]\n"
            "FACT: Saturday telehealth is not currently offered; a pilot is under discussion. [telehealth]\n"
            "FACT: Appointments can be booked through the online portal or by phone at 555-0142. [booking]"
        )

    @staticmethod
    def _write(text: str) -> str:
        return (
            "## Current hours and access\n"
            "Riverside Family Clinic is open Monday-Friday 8:00 AM-6:00 PM and Saturday 9:00 AM-1:00 PM. [clinic-hours]\n\n"
            "## Telehealth today and the Saturday gap\n"
            "Telehealth runs Monday-Friday 9:00 AM-5:00 PM for follow-ups, medication refills, and minor illness. "
            "Saturday telehealth is not currently offered, though a pilot is under discussion. [telehealth]\n\n"
            "## Recommendation\n"
            "Offer Saturday telehealth 9:00 AM-1:00 PM, matching Saturday in-person hours, and book it through the "
            "existing online portal or by phone at 555-0142. [booking]"
        )

    @staticmethod
    def _review(text: str) -> str:
        return (
            "VERDICT: PASS\n"
            "Every claim in the draft is backed by the knowledge base: hours match clinic-hours, "
            "telehealth hours and the Saturday gap match telehealth, and booking matches booking. "
            "No unsupported claims found."
        )


def get_llm(live: bool, role: str = "agent"):
    """Return a ``(system, user) -> str`` callable: real LiteLLM or the stub."""
    if live:
        return chat
    stub = StubLLM(role=role)
    return stub.complete
