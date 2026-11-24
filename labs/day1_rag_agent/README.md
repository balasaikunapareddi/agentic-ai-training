# Day 1 lab (b) — Multi-tool agent with RAG

## Learning goals

- Retrieve-then-synthesise: retrieval picks the evidence, the model only
  rephrases it. The model never answers from its own memory.
- Provenance: every answer cites the document it came from.
- Why retrieval fails: keyword overlap misses paraphrases — that failure is
  what the graph in `agent-graph-rag` exists to fix.

## What you will build

`RagAgent` in `main.py`: `retrieve()` ranks chunks from the four clinic
documents in `src/agentic_ai_training/data/` by token overlap, then the
model (or the deterministic stub synthesiser) answers from the top chunks
only. Prompts live in `prompts.py`.

## Run it

```bash
python main.py            # stub mode, no key needed
python main.py --live     # real model via LiteLLM (needs GEMINI_API_KEY)
python main.py --live --question "Does the clinic accept Aetna?"
```

Demo questions cover hours, booking, and insurance — each answer should cite
its source document (`[clinic-hours]`, `[booking]`, `[insurance]`).

## Where this pattern lives in the portfolio

- `agent-graph-rag` — same grounding rule, but documents become validated
  Pydantic entities in a knowledge graph, so multi-hop questions ("which
  compound did this reaction produce, and what was measured for it?") are
  answered by traversal instead of passage retrieval.
- `agent-booking` — the clinic booking agent this knowledge base is drawn
  from: provider directory via RAG, scheduling via a second agent.

## Exercises

1. Ask "What time does the clinic close on Sundays?" — the KB says it is
   closed. Does the agent say so, or hallucinate hours? Fix the prompt if it
   hallucinates.
2. Add a fifth document (e.g. parking) and a question that needs it.
3. Change `top_k` to 1 and to 4. When does the answer get worse, and why?
