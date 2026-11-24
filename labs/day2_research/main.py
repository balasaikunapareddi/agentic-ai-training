"""Day 2 lab (a): two-agent research system.

A researcher agent gathers facts from the clinic knowledge base; a writer
agent turns those facts into a short brief. The agents are Google ADK
``LlmAgent``s wired as a ``SequentialAgent`` (see ``adk_agents.py``) — the
framework owns the handoff, so in stub mode the writer only ever sees the
researcher's output, never the raw question. (In live ADK mode the writer
shares the session history, so it does see the topic — its instruction is
to use only the researcher's findings. That is the contract to watch.) That is the pattern from
``agent-resolution-workflow`` (Google ADK multi-agent: a coordinator plus
read-only sub-agents) and from ``agent-research-pipeline`` (research, then
synthesis, with the handoff carrying the evidence).

Live mode runs the real ADK Runner. Stub mode runs the same two roles
deterministically with the same prompts, so the lab works with no key.

Run:
    python main.py            # stub mode, no key needed
    python main.py --live     # real ADK agents via LiteLLM (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_ai_training.llm import get_llm  # noqa: E402
from agentic_ai_training.retrieval import load_documents, retrieve  # noqa: E402
from prompts import RESEARCHER_PROMPT, TOPIC, WRITER_PROMPT  # noqa: E402


class ResearchTeam:
    """Two-agent team: researcher gathers facts, writer writes the brief."""

    def __init__(self, live: bool = False):
        self.live = live

    def run(self, topic: str = TOPIC) -> dict:
        """Return {"brief": ..., "facts": ...}."""
        if self.live:
            return asyncio.run(self._run_live(topic))
        return self._run_stub(topic)

    # -- stub: same roles, deterministic ------------------------------------
    def _run_stub(self, topic: str) -> dict:
        researcher = get_llm(False, role="researcher")
        writer = get_llm(False, role="writer")
        facts = researcher(RESEARCHER_PROMPT, f"Research this topic: {topic}")
        brief = writer(WRITER_PROMPT, f"Write a brief from these findings:\n{facts}")
        return {"brief": brief, "facts": facts}

    # -- live: real ADK agents ----------------------------------------------
    async def _run_live(self, topic: str) -> dict:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        from adk_agents import build_team

        team = build_team()
        session_service = InMemorySessionService()
        runner = Runner(agent=team, app_name="research_team", session_service=session_service)
        await session_service.create_session(
            app_name="research_team", user_id="trainee", session_id="research-1"
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="trainee",
            session_id="research-1",
            new_message=types.Content(parts=[types.Part(text=topic)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(p.text or "" for p in event.content.parts)
        docs = load_documents()
        facts = "\n".join(
            f"[{doc.doc_id}] {chunk}"
            for doc, chunk, _ in retrieve(topic, docs, top_k=3)
        )
        return {"brief": final_text, "facts": facts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 2 lab (a): two-agent research system")
    parser.add_argument("--live", action="store_true", help="run the real ADK agents via LiteLLM")
    parser.add_argument("--topic", default=TOPIC, help="research topic")
    args = parser.parse_args()

    team = ResearchTeam(live=args.live)
    result = team.run(args.topic)
    print(f"Day 2 lab (a) — research team ({'LIVE ADK' if args.live else 'stub'} mode)")
    print(f"Topic: {args.topic}\n")
    print("--- Research findings ---")
    print(result["facts"])
    print("\n--- Brief ---")
    print(result["brief"])
    ok = bool(result["brief"]) and "Saturday" in result["brief"] and "telehealth" in result["brief"].lower()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
