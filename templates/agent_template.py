"""Template for a new agent in this training repo.

Copy this file to ``labs/<your-lab>/agent.py`` (or keep it as a starting
point for your own project), rename ``TemplateAgent``, and fill in the
three marked sections. The loop, the tool registry, and the LiteLLM wiring
are the same as the Day 1 labs — you only write the behaviour.

Run:
    python agent_template.py            # stub mode, no key needed
    python agent_template.py --live     # real model via LiteLLM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- 1. Make the shared package importable (adjust if you move this file) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_ai_training.llm import get_llm  # noqa: E402
from agentic_ai_training.tools import run_tool  # noqa: E402

# --- 2. System prompt: keep it in its own prompts.py in a real lab. -------
# (Inline here only because this is a single-file template.)
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using tools.

You have these tools:
- calculator: evaluate an arithmetic expression, e.g. {"expression": "18 * 7 + 5"}
- word_count: count the words in a piece of text, e.g. {"text": "the quick brown fox"}

Rules:
- Reply with exactly one JSON object, nothing else.
- If a tool can answer the question, reply {"tool": "<name>", "args": {...}}.
- If no tool is needed, reply {"answer": "<your answer>"}.
- After a tool result is given to you, reply with {"answer": "<final answer>"}.
"""


class TemplateAgent:
    """A tool-using agent. Rename me, keep the loop."""

    def __init__(self, live: bool = False):
        self.live = live
        self.llm = get_llm(live)

    def run(self, question: str) -> dict:
        """Run the agent loop. Returns {"answer": ..., "steps": [...]}."""
        steps: list[dict] = []
        reply = self.llm(SYSTEM_PROMPT, f"Question: {question}")
        for _ in range(3):
            parsed = self._parse(reply)
            if "answer" in parsed:
                return {"answer": parsed["answer"], "steps": steps}
            result = run_tool(parsed.get("tool", ""), parsed.get("args", {}))
            steps.append({"tool": parsed.get("tool"), "result": result})
            reply = self.llm(
                SYSTEM_PROMPT,
                f'The tool "{parsed.get("tool")}" returned: {result}\n'
                'Reply with exactly one JSON object: {"answer": "<final answer>"}',
            )
        return {"answer": "I ran out of steps before answering.", "steps": steps}

    @staticmethod
    def _parse(reply: str) -> dict:
        """Parse one model reply. Non-dict JSON (e.g. a bare list) is an
        answer, not a crash; text that looks like a tool call but is not
        valid JSON is reported as a parse error, not echoed as an answer."""
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            start, end = reply.find("{"), reply.rfind("}")
            if start == -1 or end == -1:
                # No complete JSON object at all — but text that opens a tool
                # call is still a malformed reply, not an answer.
                if '"tool"' in reply or "'tool'" in reply:
                    return {"answer": "Error: I could not parse the model's tool call as JSON."}
                return {"answer": reply.strip()}
            try:
                parsed = json.loads(reply[start : end + 1])
            except json.JSONDecodeError:
                if '"tool"' in reply or "'tool'" in reply:
                    return {"answer": "Error: I could not parse the model's tool call as JSON."}
                return {"answer": reply.strip()}
        return parsed if isinstance(parsed, dict) else {"answer": str(parsed)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Template agent — copy me to start a new agent")
    parser.add_argument("--live", action="store_true", help="use the real model via LiteLLM")
    parser.add_argument("--question", default="What is 18 * 7 + 5?")
    args = parser.parse_args()
    result = TemplateAgent(live=args.live).run(args.question)
    for step in result["steps"]:
        print(f"  tool {step['tool']} -> {step['result']}")
    print(f"Q: {args.question}\nA: {result['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
