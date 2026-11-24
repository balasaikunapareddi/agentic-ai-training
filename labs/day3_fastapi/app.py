"""FastAPI app for the Day 3 deployment lab.

Three endpoints, one agent:
- ``GET /health`` — liveness, no model call.
- ``POST /chat`` — ask the RAG agent, get the answer as JSON.
- ``POST /chat/stream`` — same answer as server-sent events, one chunk per
  sentence, so a UI can render it as it arrives.

The app is built by :func:`create_app` so tests can construct it without
starting a server (see ``tests/test_training.py``). The agent is created
once at startup and reused across requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentic_ai_training.llm import get_llm
from agentic_ai_training.retrieval import load_documents, retrieve
from prompts import SYSTEM_PROMPT, USER_TEMPLATE


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class RagService:
    """The Day 1 RAG agent, as a service object."""

    def __init__(self, live: bool = False):
        self.live = live
        self.llm = get_llm(live)
        self.docs = load_documents()

    def answer(self, question: str) -> ChatResponse:
        hits = retrieve(question, self.docs, top_k=2)
        if not hits:
            return ChatResponse(
                answer="I couldn't find that in the clinic knowledge base.",
                sources=[],
            )
        context = "\n\n".join(f"[{doc.doc_id}] {chunk}" for doc, chunk, _ in hits)
        sources = sorted({doc.doc_id for doc, _, _ in hits})
        user = USER_TEMPLATE.format(question=question, context=context, sources=", ".join(sources))
        text = self.llm(SYSTEM_PROMPT, user)
        # Stub synthesis already carries its source tag from the chunk;
        # live mode is instructed by the prompt to end with the source id.
        return ChatResponse(answer=text, sources=sources)


def create_app(live: bool = False) -> FastAPI:
    service = RagService(live=live)
    app = FastAPI(title="agentic-ai-training — Day 3 clinic agent")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "live" if live else "stub"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        return service.answer(request.question)

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        result = service.answer(request.question)

        def chunks():
            for sentence in result.answer.split(". "):
                sentence = sentence.strip()
                if sentence:
                    yield f"data: {sentence}.\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    return app
