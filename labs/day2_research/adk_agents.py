"""Google ADK definitions for the Day 2 research lab.

Two ``LlmAgent``s — a researcher with a search tool and a writer with none —
wired as a ``SequentialAgent``: research finishes before writing starts, so
the writer can only use what the researcher found. The model is LiteLLM
(``LiteLlm``), so the model id comes from ``LLM_MODEL`` like everywhere else
in this repo; ADK itself never sees an API key directly.

These definitions are used ONLY in ``--live`` mode. Stub mode runs the same
two roles deterministically in ``main.py`` (same prompts, same tools), so the
lab works with no key. That split is deliberate: the ADK graph is the thing
being taught, and the stub mirror is what makes it runnable offline.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from agentic_ai_training.retrieval import load_documents, retrieve
from prompts import RESEARCHER_PROMPT, WRITER_PROMPT

_DOCS = load_documents()


def search_clinic(query: str) -> str:
    """Search the Riverside Family Clinic knowledge base for facts about a topic.

    Args:
        query: The question or topic to search for, e.g. "Saturday telehealth".

    Returns:
        The most relevant passages, each tagged with its source document id.
    """
    hits = retrieve(query, _DOCS, top_k=3)
    if not hits:
        return "No matching passages in the clinic knowledge base."
    return "\n\n".join(f"[{doc.doc_id}] {chunk}" for doc, chunk, _ in hits)


def _model() -> LiteLlm:
    return LiteLlm(model=os.environ.get("LLM_MODEL", "gemini/gemini-3.5-flash-lite"))


def build_team() -> SequentialAgent:
    """Researcher (with search tool) then writer (no tools)."""
    researcher = LlmAgent(
        name="researcher",
        model=_model(),
        instruction=RESEARCHER_PROMPT,
        tools=[FunctionTool(search_clinic)],
    )
    writer = LlmAgent(
        name="writer",
        model=_model(),
        instruction=WRITER_PROMPT,
    )
    return SequentialAgent(
        name="research_team",
        sub_agents=[researcher, writer],
        description="Research a clinic question, then write a short cited brief.",
    )
