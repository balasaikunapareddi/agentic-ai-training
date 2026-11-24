# Day 2 lab (b) — Four-agent collaborative system

## Learning goals

- Planning before doing: the planner's outline constrains what the writer
  may write, so the brief has a shape before any facts exist.
- Review as a separate agent: the reviewer checks claims against the
  researcher's facts and can send the draft back — verification is a job,
  not a hope.
- Loop bounds in code: at most 2 fix rounds, enforced by a number in
  `main.py`, never by asking the model nicely.

## What you will build

`CollaborativeTeam` in `main.py`: planner → researcher → writer (a
`SequentialAgent` in `adk_agents.py`), then the reviewer runs separately;
on `NEEDS-FIX` the writer re-runs with the feedback, up to
`MAX_REVIEW_ROUNDS = 2`. Stub mode mirrors all four roles deterministically.

## Run it

```bash
python main.py            # stub mode, no key needed
python main.py --live     # real ADK agents via LiteLLM (needs GEMINI_API_KEY)
```

The stub run passes review on the first round — the stub writer only uses
stub facts. In live mode, watch whether the reviewer catches anything; a
`NEEDS-FIX` round is the lab working, not failing.

## Where this pattern lives in the portfolio

- `agent-eval-levels` — the same idea as an evaluation ladder: a draft is
  not "done" when the model says so, but when a separate check passes.
- `agent-resolution-workflow` — a deterministic workflow validates a
  resolution before anything is written; the reviewer here is that
  validation step, as an agent.
- `agent-red-team-tester` — an attacker agent and a judge agent in the same
  loop shape: one produces, the other verifies.

## Exercises

1. Make the reviewer stricter: it must also flag any claim with no citation.
   Does the writer fix it within 2 rounds?
2. Swap the order: researcher before planner. What breaks, and why does
   planning first matter?
3. Add a fifth agent — a fact-checker that re-searches the KB for each claim
   instead of trusting the researcher's list. When is that worth the cost?
