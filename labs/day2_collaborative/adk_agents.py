"""Google ADK definitions for the Day 2 collaborative lab.

Four ``LlmAgent``s — planner, researcher (with the search tool), writer, and
reviewer (no tools; it judges, it does not fetch). The first three run as a
``SequentialAgent``; the reviewer runs separately because the review loop —
re-run the writer with feedback, at most twice — is orchestrated in Python
in ``main.py``, not inside ADK. A loop bound belongs in code, where you can
see it and test it, not in a prompt.

Used ONLY in ``--live`` mode; stub mode mirrors the four roles
deterministically in ``main.py``.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

from agentic_ai_training.retrieval import load_documents, retrieve
from prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT

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
    """Planner, then researcher, then writer."""
    planner = LlmAgent(name="planner", model=_model(), instruction=PLANNER_PROMPT)
    researcher = LlmAgent(
        name="researcher",
        model=_model(),
        instruction=RESEARCHER_PROMPT,
        tools=[FunctionTool(search_clinic)],
    )
    writer = LlmAgent(name="writer", model=_model(), instruction=WRITER_PROMPT)
    return SequentialAgent(
        name="collaborative_team",
        sub_agents=[planner, researcher, writer],
        description="Plan, research, then write a brief about a clinic question.",
    )


def build_reviewer() -> LlmAgent:
    """Reviewer: judges the draft against the facts, writes no new facts."""
    return LlmAgent(name="reviewer", model=_model(), instruction=REVIEWER_PROMPT)


def build_writer() -> LlmAgent:
    """Standalone writer, used for fix rounds after a NEEDS-FIX verdict."""
    return LlmAgent(name="writer", model=_model(), instruction=WRITER_PROMPT)
