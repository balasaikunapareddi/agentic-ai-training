"""Day 1 lab (a): a basic conversational agent with tools.

The smallest real agent loop: ask the model what to do next (as JSON),
run the tool it chose, feed the result back, and stop when it answers.
The loop is 40 lines and framework-free on purpose — this is the shape
every agent in Bala's portfolio starts from (see ``agent-support-resolution``
and ``agent-graph-rag`` for the same loop grown up with policy checks and
typed tools).

Run:
    python main.py            # stub mode, no key needed
    python main.py --live     # real model via LiteLLM (needs GEMINI_API_KEY)
    python main.py --live --question "What is 12 * 9?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_ai_training.llm import get_llm  # noqa: E402
from agentic_ai_training.tools import TOOLS, run_tool  # noqa: E402
from prompts import FOLLOW_UP_PROMPT, SYSTEM_PROMPT  # noqa: E402

MAX_STEPS = 3


class BasicAgent:
    """A tool-using agent: one LLM call per step, tools dispatched by name."""

    def __init__(self, live: bool = False):
        self.live = live
        self.llm = get_llm(live)

    def run(self, question: str) -> dict:
        """Run the agent loop. Returns {"answer": ..., "steps": [...]}."""
        steps: list[dict] = []
        reply = self._ask(SYSTEM_PROMPT, f"Question: {question}")
        for _ in range(MAX_STEPS):
            parsed = self._parse(reply)
            if "answer" in parsed:
                return {"answer": parsed["answer"], "steps": steps}
            tool, args = parsed.get("tool"), parsed.get("args", {})
            result = run_tool(tool, args) if tool else "Error: no tool named"
            steps.append({"tool": tool, "args": args, "result": result})
            reply = self._ask(
                SYSTEM_PROMPT,
                FOLLOW_UP_PROMPT.format(tool=tool, result=result),
            )
        return {"answer": "I ran out of steps before answering.", "steps": steps}

    def _ask(self, system: str, user: str) -> str:
        return self.llm(system, user)

    @staticmethod
    def _parse(reply: str) -> dict:
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
                # Text that looks like a tool call but is not valid JSON is
                # a malformed model reply, not an answer: say so loudly
                # instead of handing the raw JSON back as the answer.
                if '"tool"' in reply or "'tool'" in reply:
                    return {"answer": "Error: I could not parse the model's tool call as JSON."}
                return {"answer": reply.strip()}
        return parsed if isinstance(parsed, dict) else {"answer": str(parsed)}


DEMO_QUESTIONS = [
    "What is 18 * 7 + 5?",
    "How many words are in 'the quick brown fox jumps'?",
    "What is the capital of France?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 1 lab (a): basic tool-using agent")
    parser.add_argument("--live", action="store_true", help="use the real model via LiteLLM")
    parser.add_argument("--question", help="ask one question instead of running the demo")
    args = parser.parse_args()

    agent = BasicAgent(live=args.live)
    questions = [args.question] if args.question else DEMO_QUESTIONS
    print(f"Day 1 lab (a) — basic agent ({'LIVE' if args.live else 'stub'} mode)")
    print(f"Tools available: {', '.join(sorted(TOOLS))}\n")
    ok = True
    for question in questions:
        result = agent.run(question)
        for step in result["steps"]:
            print(f"  tool {step['tool']}({step['args']}) -> {step['result']}")
        print(f"Q: {question}\nA: {result['answer']}\n")
        if not result["answer"] or result["answer"].startswith("I ran out"):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
