# Templates

## `agent_template.py` — start a new agent from this file

```bash
cp templates/agent_template.py labs/my_new_agent/agent.py
```

Then fill in the three marked sections:

1. **Imports / path** — the `sys.path` line assumes the layout of this repo
   (`src/` next to `labs/`). If you copy the template into your own project,
   point it at your own package or install the package with `pip install -e .`.
2. **System prompt** — move it into a `prompts.py` next to your agent file
   (the standing rule in this repo: prompts live in their own file, never
   inline). The template keeps it inline only because it is a single file.
3. **Tools** — add functions to `src/agentic_ai_training/tools.py` and
   register them in `TOOLS`. The loop dispatches by name, so a new tool needs
   no loop change — but the prompt must describe it, or the model will never
   call it.

## Checklist for a new agent (the bar every lab in this repo meets)

- [ ] System prompt in a separate `prompts.py`
- [ ] One visible agent class with a `run()` method
- [ ] Model from `LLM_MODEL`, key from `GEMINI_API_KEY`, via LiteLLM
- [ ] Stub mode: runs with no key and gives deterministic answers
- [ ] A `README.md` with learning goals, a one-command run, and the
      portfolio repo the pattern comes from
- [ ] A `requirements.txt` and a `.env.example`
- [ ] At least one test in `tests/` that runs the agent in stub mode
- [ ] A live run (`--live`) verified by hand before you claim it works
