"""Smoke test: run every lab's main.py in stub mode and assert exit 0.

Usage:
    python scripts/smoke_test.py            # uses .venv/bin/python if present
    python scripts/smoke_test.py --python /path/to/python

Day 3 labs are servers, so they are exercised through their app factories
with FastAPI's TestClient instead of by starting uvicorn — same code path,
no port needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = [
    "labs/day1_basic_agent",
    "labs/day1_rag_agent",
    "labs/day2_research",
    "labs/day2_collaborative",
]


def run_lab(python: str, lab: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [python, "main.py"],
        cwd=ROOT / lab,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = (proc.stdout + proc.stderr).strip().splitlines()
    tail = "\n".join(output[-6:])
    return proc.returncode == 0, f"exit={proc.returncode}\n{tail}"


def check_day3(python: str) -> tuple[bool, str]:
    """Exercise both Day 3 apps through TestClient (no server, no port)."""
    code = """
import sys
sys.path.insert(0, "labs/day3_fastapi")
sys.path.insert(0, "labs/day3_production")
sys.path.insert(0, "src")
from fastapi.testclient import TestClient
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

a = load("d3a_app", "labs/day3_fastapi/app.py")
c = TestClient(a.create_app())
assert c.get("/health").json()["status"] == "ok"
r = c.post("/chat", json={"question": "What are the clinic opening hours?"})
assert r.status_code == 200 and "8:00 AM" in r.json()["answer"], r.text
s = c.post("/chat/stream", json={"question": "How do I book an appointment?"})
assert s.status_code == 200 and "[DONE]" in s.text

b = load("d3b_app", "labs/day3_production/app.py")
c2 = TestClient(b.create_app())
assert c2.get("/health").json()["status"] == "ok"
assert c2.post("/chat", json={"question": "x"}).status_code == 401
h = {"X-API-Key": "training-demo-key"}
q = {"question": "What are the clinic opening hours?"}
r1 = c2.post("/chat", json=q, headers=h)
assert r1.status_code == 200 and r1.json()["cached"] is False
r2 = c2.post("/chat", json=q, headers=h)
assert r2.json()["cached"] is True, "second identical question must be cached"
m = c2.get("/metrics").json()
assert m["cache_hits"] >= 1 and m["requests"] >= 2, m
print("day3_fastapi: health/chat/stream OK")
print("day3_production: auth/cache/metrics OK", m)
"""
    proc = subprocess.run([python, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return proc.returncode == 0, f"exit={proc.returncode}\n{(proc.stdout + proc.stderr).strip()[-800:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test every lab in stub mode")
    parser.add_argument("--python", default=None)
    args = parser.parse_args()
    venv_python = ROOT / ".venv" / "bin" / "python"
    python = args.python or (str(venv_python) if venv_python.exists() else sys.executable)

    results: dict[str, bool] = {}
    for lab in LABS:
        ok, detail = run_lab(python, lab)
        results[lab] = ok
        print(f"[{'PASS' if ok else 'FAIL'}] {lab}\n{detail}\n")
    ok, detail = check_day3(python)
    results["labs/day3_fastapi + labs/day3_production"] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] labs/day3_fastapi + labs/day3_production\n{detail}\n")

    passed = sum(results.values())
    print(f"Smoke test: {passed}/{len(results)} lab groups passed (stub mode, no key)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
