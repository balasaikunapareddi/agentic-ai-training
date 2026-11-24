"""Deterministic tools shared by the labs.

Tools are plain typed Python functions with a small registry. The agent loop
in each lab asks the model to emit ``{"tool": name, "args": {...}}`` and then
dispatches through :func:`run_tool`. Keeping tools as ordinary functions —
not hidden inside a framework — is the point of Day 1: you can see exactly
what the model can and cannot do.
"""

from __future__ import annotations

import re


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression (numbers, + - * / and parens)."""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression or ""):
        return "Error: expression may only contain numbers, spaces, and + - * / ( )"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - input is regex-validated
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def word_count(text: str) -> str:
    """Count the words in *text*."""
    return str(len((text or "").split()))


TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Evaluate an arithmetic expression, e.g. {'expression': '18 * 7 + 5'}",
        "args": {"expression": "string, numbers with + - * / and parentheses"},
    },
    "word_count": {
        "function": word_count,
        "description": "Count the words in a piece of text, e.g. {'text': 'the quick brown fox'}",
        "args": {"text": "string to count the words of"},
    },
}


def run_tool(name: str, args: dict) -> str:
    """Dispatch a tool call by name. Unknown tools return an error string."""
    entry = TOOLS.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(TOOLS))}"
    try:
        return str(entry["function"](**args))
    except TypeError as exc:
        return f"Error: bad arguments for tool '{name}': {exc}"
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the agent loop
        return f"Error: tool '{name}' failed: {exc}"
