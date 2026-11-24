"""Day 2 lab (b): four-agent collaborative system with planning + review.

Planner outlines the brief, researcher gathers cited facts, writer drafts
it, and a reviewer checks every claim against the facts — sending the draft
back for a fix at most twice. The review loop is the lesson: verification is
a separate agent with a separate prompt, and the loop bound lives in code.

Live mode runs real Google ADK agents (``adk_agents.py``) through the ADK
Runner. Stub mode runs the same four roles deterministically, so the lab
works with no key.

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
from prompts import (  # noqa: E402
    MAX_REVIEW_ROUNDS,
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    TOPIC,
    WRITER_PROMPT,
)


class CollaborativeTeam:
    """Planner -> researcher -> writer -> reviewer (up to 2 fix rounds)."""

    def __init__(self, live: bool = False):
        self.live = live

    def run(self, topic: str = TOPIC) -> dict:
        """Return {"brief", "outline", "facts", "verdict", "rounds"}."""
        if self.live:
            return asyncio.run(self._run_live(topic))
        return self._run_stub(topic)

    # -- stub: same roles, deterministic ------------------------------------
    def _run_stub(self, topic: str) -> dict:
        planner = get_llm(False, role="planner")
        researcher = get_llm(False, role="researcher")
        writer = get_llm(False, role="writer")
        reviewer = get_llm(False, role="reviewer")
        outline = planner(PLANNER_PROMPT, f"Plan a brief on this topic: {topic}")
        facts = researcher(RESEARCHER_PROMPT, f"Research this topic: {topic}")
        brief = writer(WRITER_PROMPT, f"Outline:\n{outline}\n\nFindings:\n{facts}")
        rounds = 0
        review = reviewer(REVIEWER_PROMPT, f"Facts:\n{facts}\n\nDraft:\n{brief}")
        verdict = "PASS" if "VERDICT: PASS" in review else "NEEDS-FIX"
        while verdict == "NEEDS-FIX" and rounds < MAX_REVIEW_ROUNDS:
            rounds += 1
            brief = writer(
                WRITER_PROMPT,
                f"Outline:\n{outline}\n\nFindings:\n{facts}\n\nReviewer feedback:\n{review}",
            )
            review = reviewer(REVIEWER_PROMPT, f"Facts:\n{facts}\n\nDraft:\n{brief}")
            verdict = "PASS" if "VERDICT: PASS" in review else "NEEDS-FIX"
        return {
            "brief": brief,
            "outline": outline,
            "facts": facts,
            "verdict": verdict,
            "rounds": rounds,
            "review": review,
        }

    # -- live: real ADK agents ----------------------------------------------
    async def _run_live(self, topic: str) -> dict:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        from adk_agents import build_reviewer, build_team, build_writer

        session_service = InMemorySessionService()
        team = build_team()
        team_runner = Runner(agent=team, app_name="collab_team", session_service=session_service)
        reviewer = build_reviewer()
        review_runner = Runner(agent=reviewer, app_name="collab_team", session_service=session_service)
        writer_runner = Runner(agent=build_writer(), app_name="collab_team", session_service=session_service)

        async def run_agent(runner, session_id, message):
            await session_service.create_session(
                app_name="collab_team", user_id="trainee", session_id=session_id
            )
            final_text = ""
            async for event in runner.run_async(
                user_id="trainee",
                session_id=session_id,
                new_message=types.Content(parts=[types.Part(text=message)]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(p.text or "" for p in event.content.parts)
            return final_text

        draft = await run_agent(team_runner, "draft-1", topic)
        # The team output is the writer's draft; recover the facts for review
        # from the same retrieval the researcher used, so the reviewer judges
        # against evidence, not against the draft's own claims.
        from agentic_ai_training.retrieval import load_documents, retrieve

        facts = "\n".join(
            f"[{doc.doc_id}] {chunk}"
            for doc, chunk, _ in retrieve(topic, load_documents(), top_k=3)
        )
        review = await run_agent(
            review_runner, "review-1", f"Facts:\n{facts}\n\nDraft:\n{draft}"
        )
        verdict = "PASS" if "VERDICT: PASS" in review else "NEEDS-FIX"
        rounds = 0
        while verdict == "NEEDS-FIX" and rounds < MAX_REVIEW_ROUNDS:
            rounds += 1
            draft = await run_agent(
                writer_runner,
                f"fix-{rounds}",
                f"Rewrite this draft, fixing the reviewer feedback and nothing else.\n\nFacts:\n{facts}\n\nDraft:\n{draft}\n\nReviewer feedback:\n{review}",
            )
            review = await run_agent(
                review_runner, f"review-{rounds + 1}", f"Facts:\n{facts}\n\nDraft:\n{draft}"
            )
            verdict = "PASS" if "VERDICT: PASS" in review else "NEEDS-FIX"
        return {
            "brief": draft,
            "outline": "(see draft headings)",
            "facts": facts,
            "verdict": verdict,
            "rounds": rounds,
            "review": review,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 2 lab (b): four-agent collaborative system")
    parser.add_argument("--live", action="store_true", help="run the real ADK agents via LiteLLM")
    parser.add_argument("--topic", default=TOPIC, help="brief topic")
    args = parser.parse_args()

    team = CollaborativeTeam(live=args.live)
    result = team.run(args.topic)
    print(f"Day 2 lab (b) — collaborative team ({'LIVE ADK' if args.live else 'stub'} mode)")
    print(f"Topic: {args.topic}\n")
    print("--- Outline ---")
    print(result["outline"])
    print("\n--- Brief ---")
    print(result["brief"])
    print(f"\n--- Review: {result['verdict']} (fix rounds: {result['rounds']}) ---")
    print(result["review"])
    ok = bool(result["brief"]) and result["verdict"] in ("PASS", "NEEDS-FIX")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
