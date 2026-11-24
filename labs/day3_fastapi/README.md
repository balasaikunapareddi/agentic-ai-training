# Day 3 lab (a) — FastAPI deployment with streaming

## Learning goals

- Deployment changes the transport, not the agent: the Day 1 RAG agent runs
  unchanged behind HTTP endpoints.
- Streaming: long answers should arrive sentence by sentence (server-sent
  events), so a UI can render while the model is still thinking.
- App factories: `create_app()` builds the app so tests can call it without
  starting a server.

## What you will build

`app.py`: `GET /health`, `POST /chat` (JSON answer + sources), and
`POST /chat/stream` (SSE, one chunk per sentence, ending with `[DONE]`).
`main.py` serves it with uvicorn on port 8000.

## Run it

```bash
python main.py                  # stub mode, no key needed
python main.py --live           # real model behind the API (needs GEMINI_API_KEY)
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "What are the clinic opening hours?"}'
curl -N -X POST http://127.0.0.1:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"question": "How do I book an appointment?"}'
```

## Where this pattern lives in the portfolio

- `agent-booking` — the same shape in production: a FastAPI service on
  Render serving the booking agent over HTTP, with the model behind an
  adapter so the transport never knows which provider answers.
- `agent-mcp-apps` / `agent-a2ui` — the same agent with an interactive UI
  instead of plain text: the agent returns structured UI, not prose.

## Exercises

1. Add request logging: method, path, and milliseconds per request.
2. Add a `GET /docs` check — FastAPI gives you OpenAPI docs for free. What
   does the schema say the `/chat` response looks like?
3. Time a `/chat` call in stub mode vs `--live`. Where does the time go?
