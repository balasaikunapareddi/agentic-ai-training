"""Day 1 lab (b): multi-tool agent with RAG.

Retrieve-then-synthesise over a small clinic knowledge base: keyword
retrieval picks the top chunks, and the model answers ONLY from those chunks,
citing the source document. The retrieval here is deliberately plain
token-overlap — no embeddings, no vector DB — so the step you are learning
is visible. The pattern (grounded answers with provenance) is the one from
``agent-graph-rag``, minus the graph: there the same questions are answered
by traversing validated entities instead of retrieving passages.

Run:
    python main.py            # stub mode, no key needed
    python main.py --live     # real model via LiteLLM (needs GEMINI_API_KEY)
    python main.py --live --question "Does the clinic accept Aetna?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_ai_training.llm import get_llm  # noqa: E402
from agentic_ai_training.retrieval import load_documents, retrieve  # noqa: E402
from prompts import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402


class RagAgent:
    """Retrieve chunks, then synthesise an answer grounded in them."""

    def __init__(self, live: bool = False, top_k: int = 2):
        self.live = live
        self.top_k = top_k
        self.llm = get_llm(live)
        self.docs = load_documents()

    def run(self, question: str) -> dict:
        """Return {"answer": ..., "sources": [...], "chunks": [...]}."""
        hits = retrieve(question, self.docs, top_k=self.top_k)
        if not hits:
            return {
                "answer": "I couldn't find that in the clinic knowledge base.",
                "sources": [],
                "chunks": [],
            }
        context = "\n\n".join(f"[{doc.doc_id}] {chunk}" for doc, chunk, _ in hits)
        sources = sorted({doc.doc_id for doc, _, _ in hits})
        user = USER_TEMPLATE.format(
            question=question, context=context, sources=", ".join(sources)
        )
        answer = self.llm(SYSTEM_PROMPT, user)
        # In stub mode the synthesised sentence already carries its source
        # tag (e.g. "[insurance] ...") from the retrieved chunk; in live
        # mode the prompt asks the model to end with the source id.
        return {"answer": answer, "sources": sources, "chunks": [c for _, c, _ in hits]}


DEMO_QUESTIONS = [
    "What are the clinic's opening hours?",
    "How do I book an appointment?",
    "Does the clinic accept UnitedHealthcare?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 1 lab (b): multi-tool agent with RAG")
    parser.add_argument("--live", action="store_true", help="use the real model via LiteLLM")
    parser.add_argument("--question", help="ask one question instead of running the demo")
    args = parser.parse_args()

    agent = RagAgent(live=args.live)
    questions = [args.question] if args.question else DEMO_QUESTIONS
    print(f"Day 1 lab (b) — RAG agent ({'LIVE' if args.live else 'stub'} mode)")
    print(f"Knowledge base: {len(agent.docs)} documents\n")
    ok = True
    for question in questions:
        result = agent.run(question)
        print(f"Q: {question}")
        print(f"  sources: {', '.join(result['sources']) or 'none'}")
        print(f"A: {result['answer']}\n")
        if not result["answer"]:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
