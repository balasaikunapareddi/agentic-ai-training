# Day 3 lab (b) — Production-ready API

## Learning goals

- The four things a demo API lacks: auth, caching, monitoring, rate limiting
  — and where each one lives (middleware, dependency, endpoint, counter).
- Caching by question: identical questions never re-call the model. Watch
  `/metrics` to prove it.
- Honest limits: this is a *local demo* of production shape. The cache and
  rate limiter are in-memory (they reset on restart), the key is a shared
  demo string, and metrics are counters, not Prometheus.

## What you will build

`app.py` wraps the Day 3 (a) `RagService` with:

- `X-API-Key` auth (`TRAINING_API_KEY`, default `training-demo-key`)
- an in-memory answer cache (repeat questions return `cached: true`)
- `GET /metrics` — requests, cache hits, errors, rate-limited count, average latency
- request-id + latency logging on every request
- 10 requests/minute per key, then HTTP 429

## Run it

```bash
python main.py                  # stub mode on http://127.0.0.1:8001
python main.py --live           # real model behind the API (needs GEMINI_API_KEY)
```

```bash
curl http://127.0.0.1:8001/metrics
curl -X POST http://127.0.0.1:8001/chat -H 'Content-Type: application/json' \
  -H 'X-API-Key: training-demo-key' \
  -d '{"question": "What are the clinic opening hours?"}'
# Ask again: cached: true. Omit the key: 401. 11th request in a minute: 429.
```

## What a real deployment adds

- Per-user auth (OAuth/JWT), not a shared key; secrets in a vault, never in code.
- Redis for the cache and the rate limiter, so they survive restarts and scale past one process.
- Prometheus + alerts on latency/error rate; structured logs shipped to a collector.
- Timeouts and retries around the model call, and a circuit breaker when the provider is down.
- Tests that hit the API the way this lab's `tests/test_training.py` does — with `TestClient`, no server needed.

## Exercises

1. Change the rate limit to 5/minute and prove the 6th request gets 429.
2. Add a `POST /cache/clear` endpoint (auth required). When would an operator need it?
3. Cache answers for only 60 seconds (store the time with the answer). Why is a stale clinic-hours answer worse than a stale joke answer?
