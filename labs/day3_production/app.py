"""Production-shaped FastAPI app for the Day 3 production lab.

Same RAG agent as lab (a), plus the four things a demo API lacks:
- **Auth**: ``X-API-Key`` header, checked against ``TRAINING_API_KEY``
  (default ``training-demo-key`` — change it; this is a local demo).
- **Caching**: identical questions return the cached answer without calling
  the model again (cache hits are counted in ``/metrics``).
- **Monitoring**: ``GET /metrics`` reports request counts, cache hits,
  errors, and average latency. Every request is logged with a request id
  and its latency in milliseconds.
- **Rate limiting**: 10 requests per minute per API key (in-memory counter).
  The 11th gets HTTP 429.

All four are deliberately small and in-memory: the point is to see where
each concern lives, not to ship a real gateway. A real deployment would
use a proper auth provider, Redis, and a metrics backend — see the README.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

import importlib.util

# Reuse the Day 3 (a) RagService, loaded by file path: both labs have an
# app.py, so a plain `from app import ...` would be ambiguous.
_DAY3A_DIR = Path(__file__).resolve().parent.parent / "day3_fastapi"
sys.path.insert(0, str(_DAY3A_DIR))  # lets the Day 3 (a) module find its prompts.py
_spec = importlib.util.spec_from_file_location("day3_fastapi_app", _DAY3A_DIR / "app.py")
_day3a = importlib.util.module_from_spec(_spec)
sys.modules.pop("prompts", None)  # the Day 3 (a) app has its own prompts.py
_spec.loader.exec_module(_day3a)
RagService = _day3a.RagService

API_KEY_NAME = "X-API-Key"
RATE_LIMIT_PER_MINUTE = 10

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool = False


class Metrics:
    """In-memory counters. A real deployment would export these to Prometheus."""

    def __init__(self):
        self.requests = 0
        self.cache_hits = 0
        self.errors = 0
        self.rate_limited = 0
        self.total_latency_ms = 0.0
        self._calls: dict[str, list[float]] = defaultdict(list)

    def snapshot(self) -> dict:
        avg = self.total_latency_ms / self.requests if self.requests else 0.0
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "rate_limited": self.rate_limited,
            "avg_latency_ms": round(avg, 2),
        }

    def check_rate_limit(self, key: str) -> bool:
        """True if the key is within its per-minute budget."""
        now = time.time()
        calls = [t for t in self._calls[key] if now - t < 60]
        if len(calls) >= RATE_LIMIT_PER_MINUTE:
            self._calls[key] = calls
            return False
        calls.append(now)
        self._calls[key] = calls
        return True


def expected_api_key() -> str:
    return os.environ.get("TRAINING_API_KEY", "training-demo-key")


def create_app(live: bool = False) -> FastAPI:
    service = RagService(live=live)
    metrics = Metrics()
    cache: dict[str, ChatResponse] = {}
    app = FastAPI(title="agentic-ai-training — Day 3 production clinic agent")
    app.state.metrics = metrics

    def require_api_key(key: str | None = Depends(api_key_header)) -> str:
        if key != expected_api_key():
            metrics.errors += 1
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return key

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        started = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - started) * 1000
        metrics.requests += 1
        metrics.total_latency_ms += latency_ms
        print(f"[{request_id}] {request.method} {request.url.path} "
              f"-> {response.status_code} ({latency_ms:.1f} ms)")
        response.headers["X-Request-Id"] = request_id
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "live" if live else "stub"}

    @app.get("/metrics")
    def get_metrics():
        return metrics.snapshot()

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, key: str = Depends(require_api_key)):
        if not metrics.check_rate_limit(key):
            metrics.rate_limited += 1
            raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests/minute")
        question = request.question.strip()
        if question in cache:
            metrics.cache_hits += 1
            hit = cache[question]
            return ChatResponse(answer=hit.answer, sources=hit.sources, cached=True)
        result = service.answer(question)
        response = ChatResponse(answer=result.answer, sources=result.sources, cached=False)
        cache[question] = response
        return response

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest, key: str = Depends(require_api_key)):
        if not metrics.check_rate_limit(key):
            metrics.rate_limited += 1
            raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests/minute")
        result = service.answer(request.question)

        def chunks():
            for sentence in result.answer.split(". "):
                sentence = sentence.strip()
                if sentence:
                    yield f"data: {sentence}.\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    return app
