"""Tests for agentic-ai-training. Everything here runs in stub mode (no key).

Lab mains are loaded by file path (each lab dir is added to sys.path so its
``from prompts import ...`` resolves), exactly as a trainee would run them.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_ai_training.retrieval import load_documents, retrieve  # noqa: E402
from agentic_ai_training.tools import calculator, run_tool, word_count  # noqa: E402


def load_lab_module(name: str, lab: str, filename: str = "main.py"):
    lab_dir = ROOT / "labs" / lab
    sys.path.insert(0, str(lab_dir))
    # Every lab has its own prompts.py; evict the cached one so this lab's
    # `from prompts import ...` binds to the right file.
    sys.modules.pop("prompts", None)
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, lab_dir / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- tools -----------------------------------------------------------------

def test_calculator():
    assert calculator("18 * 7 + 5") == "131"
    assert calculator("(10 + 2) / 3") == "4"
    assert calculator("2 + two") == (
        "Error: expression may only contain numbers, spaces, and + - * / ( )"
    )


def test_word_count():
    assert word_count("the quick brown fox jumps") == "5"


def test_unknown_tool_errors():
    assert "unknown tool" in run_tool("teleport", {})


# --- retrieval ---------------------------------------------------------------

def test_retrieval_finds_hours_and_booking():
    docs = load_documents()
    assert len(docs) == 4
    top = retrieve("What are the clinic's opening hours?", docs, top_k=1)[0]
    assert top[0].doc_id == "clinic-hours"
    assert "8:00 AM" in top[1]
    top = retrieve("How do I book an appointment?", docs, top_k=1)[0]
    assert top[0].doc_id == "booking"
    assert "555-0142" in top[1]


# --- Day 1 labs (stub) -------------------------------------------------------

def test_day1_basic_agent_stub():
    mod = load_lab_module("d1a_main", "day1_basic_agent")
    agent = mod.BasicAgent(live=False)
    result = agent.run("What is 18 * 7 + 5?")
    assert "131" in result["answer"] and len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "calculator"
    result = agent.run("How many words are in 'the quick brown fox jumps'?")
    assert "5" in result["answer"] and len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "word_count"
    assert "Paris" in agent.run("What is the capital of France?")["answer"]


def test_day1_rag_agent_stub():
    mod = load_lab_module("d1b_main", "day1_rag_agent")
    agent = mod.RagAgent(live=False)
    hours = agent.run("What are the clinic's opening hours?")
    assert "8:00 AM" in hours["answer"] and "clinic-hours" in hours["sources"]
    booking = agent.run("How do I book an appointment?")
    assert "555-0142" in booking["answer"] and "booking" in booking["sources"]
    insurance = agent.run("Does the clinic accept UnitedHealthcare?")
    assert "UnitedHealthcare" in insurance["answer"]
    unknown = agent.run("What is the meaning of life?")
    assert "couldn't find" in unknown["answer"]


# --- Day 2 labs (stub) -------------------------------------------------------

def test_day2_research_team_stub():
    mod = load_lab_module("d2a_main", "day2_research")
    result = mod.ResearchTeam(live=False).run()
    assert "Saturday" in result["brief"] and "telehealth" in result["brief"].lower()
    assert "[telehealth]" in result["brief"] and "[clinic-hours]" in result["facts"]


def test_day2_collaborative_team_stub():
    mod = load_lab_module("d2b_main", "day2_collaborative")
    result = mod.CollaborativeTeam(live=False).run()
    assert result["verdict"] == "PASS"
    assert result["rounds"] == 0
    assert "Saturday" in result["brief"]
    assert "Current hours" in result["outline"]


# --- templates ---------------------------------------------------------------

def test_template_agent_stub():
    spec = importlib.util.spec_from_file_location(
        "template_agent", ROOT / "templates" / "agent_template.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.TemplateAgent(live=False).run("What is 18 * 7 + 5?")
    assert "131" in result["answer"]


# --- Day 3 apps (TestClient, no server) --------------------------------------

def _load_app(name: str, lab: str):
    lab_dir = ROOT / "labs" / lab
    sys.path.insert(0, str(lab_dir))
    sys.modules.pop("prompts", None)
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, lab_dir / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day3_fastapi_app():
    from fastapi.testclient import TestClient

    app = _load_app("d3a_app", "day3_fastapi").create_app()
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "mode": "stub"}
    resp = client.post("/chat", json={"question": "What are the clinic opening hours?"})
    assert resp.status_code == 200
    assert "8:00 AM" in resp.json()["answer"]
    assert "clinic-hours" in resp.json()["sources"]
    stream = client.post("/chat/stream", json={"question": "How do I book an appointment?"})
    assert stream.status_code == 200
    assert "[DONE]" in stream.text and "555-0142" in stream.text


def test_day3_production_app_auth_cache_metrics_rate_limit():
    from fastapi.testclient import TestClient

    app = _load_app("d3b_app", "day3_production").create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "training-demo-key"}
    question = {"question": "What are the clinic opening hours?"}

    assert client.post("/chat", json=question).status_code == 401
    assert client.post("/chat", json=question, headers={"X-API-Key": "wrong"}).status_code == 401

    first = client.post("/chat", json=question, headers=headers)
    assert first.status_code == 200 and first.json()["cached"] is False
    second = client.post("/chat", json=question, headers=headers)
    assert second.json()["cached"] is True
    assert "X-Request-Id" in second.headers

    metrics = client.get("/metrics").json()
    assert metrics["cache_hits"] >= 1
    assert metrics["requests"] >= 2

    # 10 requests/minute per key: 2 used above, 8 more succeed, the 11th is 429.
    for i in range(8):
        assert client.post("/chat", json={"question": f"q{i}"}, headers=headers).status_code == 200
    assert client.post("/chat", json={"question": "one too many"}, headers=headers).status_code == 429
    assert client.get("/metrics").json()["rate_limited"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
