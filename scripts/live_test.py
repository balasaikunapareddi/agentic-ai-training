"""Live test: run the Day 1 labs against the real model and score them.

Run via the Gemini helper (it injects GEMINI_API_KEY; never paste a key):
    python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \\
        ~/workspace/agentic-ai-training .venv/bin/python scripts/live_test.py

Scores each lab on fixed expected answers and prints a table. Exit 0 only
if every check passes. A Gemini 503 is retried once inside the llm wrapper;
if it still fails, the failure is reported, not hidden.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")

CHECKS = [
    # (lab dir, extra args, substrings that must appear in stdout)
    ("labs/day1_basic_agent", [], ["131", "5", "Paris"]),
    ("labs/day1_rag_agent", [], ["8:00 AM", "555-0142", "UnitedHealthcare"]),
]


def main() -> int:
    print("Live test — real Gemini model (gemini-3.5-flash-lite via LiteLLM)\n")
    all_ok = True
    for lab, extra, expected in CHECKS:
        proc = subprocess.run(
            [PYTHON, "main.py", "--live", *extra],
            cwd=ROOT / lab,
            capture_output=True,
            text=True,
            timeout=300,
        )
        found = [e for e in expected if e in proc.stdout]
        missing = [e for e in expected if e not in proc.stdout]
        ok = proc.returncode == 0 and not missing
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {lab} (exit={proc.returncode})")
        print(f"  expected in output: {found}")
        if missing:
            print(f"  MISSING: {missing}")
        print("  --- output ---")
        print(proc.stdout.strip())
        if proc.stderr.strip():
            print("  --- stderr ---")
            print(proc.stderr.strip()[-1000:])
        print()
    print(f"Live test: {'ALL PASSED' if all_ok else 'FAILURES — see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
