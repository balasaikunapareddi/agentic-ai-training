"""Day 3 lab (b): production-ready API with auth/caching/monitoring.

Same RAG agent, wrapped the way a real service wraps it: API-key auth,
an answer cache, a ``/metrics`` endpoint, request-id logging, and a
per-key rate limit.

Run:
    python main.py                  # stub mode on http://127.0.0.1:8001
    python main.py --live           # real model behind the API

Try it (default demo key ``training-demo-key`` — set TRAINING_API_KEY to change it):
    curl http://127.0.0.1:8001/health
    curl http://127.0.0.1:8001/metrics
    curl -X POST http://127.0.0.1:8001/chat \\
      -H 'Content-Type: application/json' -H 'X-API-Key: training-demo-key' \\
      -d '{"question": "What are the clinic opening hours?"}'
    # Ask the same question again: the second answer is served from cache.
    # Omit the X-API-Key header: HTTP 401. Send 11 requests in a minute: HTTP 429.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import uvicorn  # noqa: E402

from app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 3 lab (b): production-ready API")
    parser.add_argument("--live", action="store_true", help="use the real model via LiteLLM")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    print(f"Day 3 lab (b) — production clinic API ({'LIVE' if args.live else 'stub'} mode)")
    print(f"Serving on http://{args.host}:{args.port} — /health, /chat, /chat/stream, /metrics")
    print("Auth: X-API-Key header (TRAINING_API_KEY, default 'training-demo-key')")
    uvicorn.run(create_app(live=args.live), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
