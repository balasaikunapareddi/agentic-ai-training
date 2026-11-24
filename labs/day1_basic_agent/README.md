# Day 1 lab (a) — Basic conversational agent with tools

## Learning goals

- The agent loop: model decides → tool runs → result goes back to the model → model answers.
- Tools as plain typed functions with a registry — the model only ever sees a name and a JSON schema.
- The same loop in stub mode and live mode: only the reasoning changes.

## What you will build

`BasicAgent` in `main.py` (~40 lines of loop). It asks the model for one JSON
object per step — either `{"tool": ..., "args": {...}}` or `{"answer": ...}` —
dispatches through `run_tool`, and stops after at most 3 steps. The system
prompt lives in `prompts.py`, never inline.

## Run it

```bash
python main.py            # stub mode, no key needed
python main.py --live     # real model via LiteLLM (needs GEMINI_API_KEY)
python main.py --live --question "What is 12 * 9?"
```

Demo questions: `18 * 7 + 5` (calculator → 131), a word count (→ 5), and
the capital of France (answered directly, no tool).

## Where this pattern lives in the portfolio

- `agent-support-resolution` — the same loop with policy-as-code guards: the
  coordinator may only call read-only tools, and writes go through a
  deterministic workflow, never the model.
- `agent-graph-rag` — the same loop with typed Pydantic tools and a
  provenance-carrying knowledge graph behind the retrieval tool.

## Exercises

1. Add a `reverse_text` tool in `src/agentic_ai_training/tools.py` and teach
   the prompt about it. The loop should not change.
2. Make the agent refuse arithmetic over 1,000,000 — a policy check in code,
   not in the prompt. (That is the policy-as-code pattern from Day 2.)
3. Swap the model: `LLM_MODEL=openai/gpt-4o-mini main.py --live` — no code change.
