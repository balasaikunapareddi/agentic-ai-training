# Day 2 lab (a) — Two-agent research system

## Learning goals

- Splitting one big job into two small agents: a researcher that only
  gathers facts, and a writer that only writes from those facts.
- The handoff is the contract: in stub mode the writer never sees the raw
  question, only the researcher's cited findings — so it cannot invent facts
  the researcher did not find. (In live ADK mode the writer shares the
  session history and does see the topic; the instruction to use only the
  researcher's findings is the contract there.)
- Google ADK as the orchestration layer: `SequentialAgent` runs the agents in
  order and passes the output along. You write agents, not plumbing.

## What you will build

`adk_agents.py` defines the team: a researcher `LlmAgent` with a
`search_clinic` tool (keyword retrieval over the clinic knowledge base) and
a writer `LlmAgent` with no tools, wired as a `SequentialAgent`. `main.py`
runs it through ADK's `Runner` in `--live` mode, and runs the same two roles
deterministically (same prompts, same tools) in stub mode.

## Run it

```bash
python main.py            # stub mode, no key needed
python main.py --live     # real ADK agents via LiteLLM (needs GEMINI_API_KEY)
python main.py --topic "Should the clinic extend Saturday hours?"
```

The brief should cover current hours, telehealth today and the Saturday gap,
and a recommendation — every claim cited to a knowledge-base document.

## Where this pattern lives in the portfolio

- `agent-resolution-workflow` — Google ADK multi-agent: a coordinator owns
  the conversation while order and policy sub-agents do read-only lookups.
- `agent-research-pipeline` — research then synthesis, with the handoff
  carrying the evidence forward.
- `agent-booking` — provider directory agent (RAG) plus scheduling agent,
  with an A2A handoff that never re-asks a question.

## Exercises

1. Add a third agent — an editor that shortens the brief to 100 words.
   Where does it go in the `SequentialAgent`?
2. Give the writer a tool (e.g. a word counter) and watch it use it — or
   not. What in the prompt decides that?
3. Ask a topic the knowledge base cannot answer ("Should the clinic open a
   second site?"). What does the researcher return, and what should the
   writer do with "no facts found"?
