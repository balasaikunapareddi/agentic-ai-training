# agentic-ai-training

**Hands-on Agentic AI training in the spirit of Vinay Kumar's 3-day course, built from real portfolio agents: day-wise labs from a basic tool-using agent and RAG, through multi-agent research and planning, to a production FastAPI deployment — every lab runnable in one command.**

## The pain

Most agent tutorials are chatbots or copy-paste snippets. You finish them
able to call an LLM, and no closer to knowing how a real agent is structured,
tested, and shipped: where the system prompt lives, how tools are typed and
guarded, how two agents hand work to each other without inventing facts, or
what stands between a demo and a service you can run. This repo teaches
exactly that, in three days, using patterns taken from real agents in
[Bala Sai Kunapareddi's portfolio](https://github.com/balasaikunapareddi) —
not toy code written for the course.

## The 3-day arc

| Day | Lab | What you learn | Pattern source |
|-----|-----|----------------|----------------|
| 1a | `labs/day1_basic_agent` | The agent loop: model decides → tool runs → result goes back | `agent-support-resolution`, `agent-graph-rag` |
| 1b | `labs/day1_rag_agent` | Multi-tool agent with RAG: retrieve, then answer only from what was retrieved, with citations | `agent-graph-rag`, `agent-booking` |
| 2a | `labs/day2_research` | Two-agent research: researcher gathers cited facts, writer writes the brief | `agent-resolution-workflow`, `agent-research-pipeline` |
| 2b | `labs/day2_collaborative` | Four-agent team: planner → researcher → writer → reviewer, with a bounded fix loop | `agent-eval-levels`, `agent-red-team-tester` |
| 3a | `labs/day3_fastapi` | FastAPI deployment with streaming (`/health`, `/chat`, `/chat/stream`) | `agent-booking` |
| 3b | `labs/day3_production` | Production shape: API-key auth, answer cache, `/metrics`, request logging, rate limiting | `agent-booking` |

Every lab is a small runnable exercise that teaches the pattern, with a
pointer to the full repo. The labs are not the repos themselves — they are
the smallest program that shows how the pattern works.

## Frameworks — why these

- **Day 1: no framework, plain Python.** The agent loop is 40 lines. You
  should see it before a framework hides it.
- **Day 2: Google ADK.** Multi-agent work is where a framework earns its
  keep: `SequentialAgent` owns the handoff, `LlmAgent` owns one job each,
  and the model is ADK's `LiteLlm` wrapper so the model id still comes from
  `LLM_MODEL`. The review loop itself stays in plain Python — a loop bound
  belongs in code, not in a prompt.
- **Day 3: FastAPI + uvicorn.** The deployment standard for the portfolio's
  Python services.
- **Everywhere: LiteLLM.** One call shape, model from the `LLM_MODEL` env
  var (default `gemini/gemini-3.5-flash-lite`), so swapping provider or
  model is a config change, not a code change.

## Prerequisites

- Python 3.11+
- No API key needed for the default stub mode. For `--live` runs, a Gemini
  API key in `GEMINI_API_KEY` (free tier).

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e .[test]
cp .env.example .env   # only needed for --live runs
```

## Run each lab in one command

```bash
.venv/bin/python labs/day1_basic_agent/main.py
.venv/bin/python labs/day1_rag_agent/main.py
.venv/bin/python labs/day2_research/main.py
.venv/bin/python labs/day2_collaborative/main.py
.venv/bin/python labs/day3_fastapi/main.py        # serves on :8000
.venv/bin/python labs/day3_production/main.py     # serves on :8001
```

Add `--live` to any of them to run the same lab against the real model
(`GEMINI_API_KEY` required). Every lab prints its demo questions and
answers; exit code 0 means it worked.

## Verify everything

```bash
.venv/bin/python scripts/smoke_test.py   # every lab in stub mode + both Day 3 apps
.venv/bin/python -m pytest tests/ -q     # unit tests, stub mode
```

Live verification (needs the key; the helper injects it, never paste one):

```bash
python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \
  ~/workspace/agentic-ai-training .venv/bin/python scripts/live_test.py
```

## Verified results (run firsthand, 2026-11-24)

| Check | Result |
|---|---|
| Unit tests (`pytest tests/`) | **11 passed** (stub mode) |
| Smoke test (`scripts/smoke_test.py`) | **5/5 lab groups passed** (stub mode, no key) |
| Live test (`scripts/live_test.py`, real Gemini `gemini-3.5-flash-lite`) | **ALL PASSED**: Day 1a 3/3 checks (131, 5 words, Paris), Day 1b 3/3 checks (8:00 AM, 555-0142, UnitedHealthcare) |
| Day 2 research live (real ADK agents) | **Passed on retry**: first run hit Gemini 503, re-run after ~3 min produced a cited 3-section brief |
| Day 3 FastAPI live (real model over HTTP) | **Passed**: `/health` ok, `/chat` returned grounded hours answer with `[clinic-hours]` citation |
| Day 2 collaborative live (real ADK agents) | **Not passed — Gemini 503 twice** (free-tier high demand, ~3 min apart). Verified in stub mode only (pytest + smoke). Re-run `--live` when the model is less loaded; no code change needed. |

Two real bugs were found and fixed while verifying: LiteLLM 1.106.2 (pinned by
`google-adk`) needs the `mcp` package installed for the ADK completion path
(added to dependencies), and the stub RAG citation appended the wrong source
document (the answer sentence already carries its own source tag — the
append was removed).

## Conventions (the standing rules, applied to every lab)

- **System prompts live in a separate `prompts.py` per lab**, never inline.
  Change behaviour by editing the prompt file, not the agent.
- **Proper package structure, visible agent classes**: `BasicAgent`,
  `RagAgent`, `ResearchTeam`, `CollaborativeTeam`, `RagService` — one class
  per lab, with a `run()` / `answer()` you can read in one screen.
- **Stub mode by default**: every lab runs offline with deterministic
  answers, so the loop, the tools, and the tests are exercisable with no
  key. Live mode changes only the reasoning, never the structure.
- **Policy in code, not in prompts**: the review loop is capped at 2
  rounds by a number in `main.py`; the tool registry only dispatches tools
  that exist.

## Honest limits

- The labs teach patterns from the portfolio repos; they are not the repos.
  The RAG lab uses keyword retrieval, not the validated knowledge graph in
  `agent-graph-rag`; the research labs use a four-document clinic knowledge
  base, not the web.
- The Day 3 production lab is a **local demo of production shape** — the
  cache and rate limiter are in-memory, the API key is a shared demo
  string, metrics are counters. A real deployment adds per-user auth,
  Redis, Prometheus, timeouts, and a circuit breaker (listed in the lab
  README).
- Stub answers are deterministic by design. If a stub answer looks too
  neat, that is the point: the stub exists to make the structure testable,
  not to pretend to be intelligent.

## Start a new agent

Copy `templates/agent_template.py`, follow the checklist in
`templates/README.md`, and add a test in `tests/`.

## Acknowledgements

The 3-day shape follows Vinay Kumar's
[Agentic_AI_Training](https://github.com/VinayKDH/Agentic_AI_Training)
course (Day 1 foundations incl. multi-tool RAG, Day 2 multi-agent research
and collaboration, Day 3 FastAPI production deployment). The labs themselves
are built from Bala's own agents listed in the table above.
