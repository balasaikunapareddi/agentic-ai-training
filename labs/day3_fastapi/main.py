"""Day 3 lab (a): FastAPI deployment with streaming.

Run the Day 1 RAG agent as an HTTP service: ``/health``, ``/chat``, and a
streaming ``/chat/stream`` (server-sent events, one chunk per sentence).

Run:
    python main.py                  # stub mode on http://127.0.0.1:8000
    python main.py --live           # real model behind the API
    python main.py --port 8001

Try it:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/chat \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "What are the clinic opening hours?"}'
    curl -N -X POST http://127.0.0.1:8000/chat/stream \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "How do I book an appointment?"}'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import uvicorn  # noqa: E402

from app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 3 lab (a): FastAPI deployment with streaming")
    parser.add_argument("--live", action="store_true", help="use the real model via LiteLLM")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Day 3 lab (a) — FastAPI clinic agent ({'LIVE' if args.live else 'stub'} mode)")
    print(f"Serving on http://{args.host}:{args.port} — /health, /chat, /chat/stream")
    uvicorn.run(create_app(live=args.live), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
