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

    metrics = client.get("/metrics", headers=headers).json()
    assert metrics["cache_hits"] >= 1
    assert metrics["requests"] >= 2

    # 10 requests/minute per key: 2 used above, 8 more succeed, the 11th is 429.
    for i in range(8):
        assert client.post("/chat", json={"question": f"q{i}"}, headers=headers).status_code == 200
    assert client.post("/chat", json={"question": "one too many"}, headers=headers).status_code == 429
    assert client.get("/metrics", headers=headers).json()["rate_limited"] >= 1


# --- review fixes (2026-11-24) -----------------------------------------------

def test_run_tool_catches_tool_exceptions():
    # A tool raising a non-TypeError used to crash the agent loop: word_count
    # with a non-string argument raised AttributeError out of run_tool.
    assert run_tool("word_count", {"text": 123}).startswith(
        "Error: tool 'word_count' failed"
    )
    assert run_tool("calculator", {"expression": 5}).startswith("Error:")


def test_rag_stub_answer_always_cites_its_source():
    # The synthesised sentence only carried its [doc-id] tag when it was the
    # first sentence of its chunk; a second-sentence answer (e.g. the
    # Sunday-closed sentence) came back with no citation at all.
    mod = load_lab_module("d1b_main", "day1_rag_agent")
    agent = mod.RagAgent(live=False)
    cases = [
        ("What are the clinic's opening hours?", "[clinic-hours]"),
        ("How do I book an appointment?", "[booking]"),
        ("Does the clinic accept UnitedHealthcare?", "[insurance]"),
        ("What time does the clinic close on Sundays?", "[clinic-hours]"),
    ]
    for question, tag in cases:
        answer = agent.run(question)["answer"]
        assert tag in answer, f"{question!r} -> {answer!r}"


def test_day1_basic_agent_malformed_tool_call():
    # A truncated tool-call JSON was echoed back as the "answer"; a bare
    # JSON list is an answer, not a crash.
    mod = load_lab_module("d1a_main", "day1_basic_agent")
    agent = mod.BasicAgent(live=False)
    agent.llm = lambda system, user: '{"tool": "calculator", "args": {"expression": "18 * 7 + 5"'
    result = agent.run("What is 18 * 7 + 5?")
    assert "could not parse" in result["answer"]
    assert "18 * 7" not in result["answer"]
    agent.llm = lambda system, user: "[1, 2]"
    assert agent.run("anything")["answer"] == "[1, 2]"


def test_template_agent_malformed_json():
    # The template crashed with AttributeError on non-dict JSON (a list has
    # no .get), and echoed truncated tool calls as answers.
    spec = importlib.util.spec_from_file_location(
        "template_agent_malformed", ROOT / "templates" / "agent_template.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agent = mod.TemplateAgent(live=False)
    agent.llm = lambda system, user: "[1, 2]"
    assert agent.run("What is 18 * 7 + 5?")["answer"] == "[1, 2]"
    agent.llm = lambda system, user: '{"tool": "calculator", "args": {"expression": "1+1"'
    assert "could not parse" in agent.run("What is 1+1?")["answer"]


def _fake_review_llm(failures: int):
    """A get_llm stand-in: real stub roles, except the reviewer says
    NEEDS-FIX `failures` times and then PASS, and the writer numbers its
    drafts so rewrites are observable."""
    from agentic_ai_training.llm import StubLLM

    state = {"reviews": 0, "drafts": 0}

    def fake_get_llm(live: bool, role: str = "agent"):
        stub = StubLLM(role=role)

        def complete(system: str, user: str) -> str:
            if role == "reviewer":
                state["reviews"] += 1
                if state["reviews"] <= failures:
                    return (
                        "VERDICT: NEEDS-FIX\nISSUE: the draft claims Saturday "
                        "telehealth hours the facts do not support."
                    )
                return stub.complete(system, user)
            if role == "writer":
                state["drafts"] += 1
                return f"Draft {state['drafts']}: brief with Saturday hours. [clinic-hours]"
            return stub.complete(system, user)

        return complete

    return fake_get_llm, state


def test_day2_collaborative_review_loop_fix_round(monkeypatch):
    # The NEEDS-FIX path was never exercised: the stub reviewer always
    # passed, so the loop bound and the rewrite were untested.
    mod = load_lab_module("d2b_main", "day2_collaborative")
    fake, state = _fake_review_llm(failures=1)
    monkeypatch.setattr(mod, "get_llm", fake)
    result = mod.CollaborativeTeam(live=False).run()
    assert result["verdict"] == "PASS"
    assert result["rounds"] == 1
    assert state["drafts"] == 2  # initial draft + one rewrite
    assert "Draft 2" in result["brief"]


def test_day2_collaborative_review_loop_max_rounds(monkeypatch):
    mod = load_lab_module("d2b_main", "day2_collaborative")
    fake, state = _fake_review_llm(failures=99)
    monkeypatch.setattr(mod, "get_llm", fake)
    result = mod.CollaborativeTeam(live=False).run()
    assert result["verdict"] == "NEEDS-FIX"
    assert result["rounds"] == mod.MAX_REVIEW_ROUNDS == 2
    assert state["drafts"] == 3  # initial draft + 2 fix rounds, then it stops


def test_day3_stream_sentences_are_not_repunctuated():
    # The old splitter did answer.split(". ") then re-added "." to every
    # chunk, so an answer ending in "." streamed as ".." ("555-0142..").
    from fastapi.testclient import TestClient

    mod = _load_app("d3a_app", "day3_fastapi")
    client = TestClient(mod.create_app())
    stream = client.post("/chat/stream", json={"question": "How do I book an appointment?"})
    assert stream.status_code == 200
    assert "555-0142." in stream.text and ".." not in stream.text
    unknown = client.post("/chat/stream", json={"question": "What is the meaning of life?"})
    assert ".." not in unknown.text
    # Helper directly: a multi-line answer stays one data line per sentence.
    chunks = list(mod.sse_chunks("First sentence.\nSecond sentence without terminator"))
    assert chunks == [
        "data: First sentence.\n\n",
        "data: Second sentence without terminator\n\n",
        "data: [DONE]\n\n",
    ]
    # A trailing [doc-id] citation stays attached to its sentence.
    assert list(mod.sse_chunks("Open 8:00 AM to 6:00 PM. [clinic-hours]")) == [
        "data: Open 8:00 AM to 6:00 PM. [clinic-hours]\n\n",
        "data: [DONE]\n\n",
    ]
    assert list(mod.sse_chunks("")) == ["data: [DONE]\n\n"]


def test_day3_production_metrics_requires_auth():
    # /metrics used to answer unauthenticated, leaking usage counters.
    from fastapi.testclient import TestClient

    app = _load_app("d3b_app", "day3_production").create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "training-demo-key"}
    assert client.get("/health").status_code == 200  # liveness stays public
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/metrics", headers=headers).status_code == 200
    assert client.post("/chat/stream", json={"question": "x"}).status_code == 401


def test_day3_production_stream_uses_cache():
    # /chat/stream used to bypass the cache entirely and re-call the model
    # on every identical question.
    from fastapi.testclient import TestClient

    app = _load_app("d3b_app", "day3_production").create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "training-demo-key"}
    question = {"question": "What are the clinic opening hours?"}
    first = client.post("/chat/stream", json=question, headers=headers)
    second = client.post("/chat/stream", json=question, headers=headers)
    assert first.status_code == 200 and second.status_code == 200
    assert first.text == second.text
    assert ".." not in second.text
    assert client.get("/metrics", headers=headers).json()["cache_hits"] >= 1


def test_day3_production_provider_failure_is_502_not_cached(monkeypatch):
    # A failing model used to surface as a bare 500, uncounted in /metrics,
    # and (for /chat) the failure path never recorded the error at all.
    from fastapi.testclient import TestClient

    mod = _load_app("d3b_app", "day3_production")

    class FailingService:
        def __init__(self, live: bool = False):
            pass

        def answer(self, question):
            raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(mod, "RagService", FailingService)
    client = TestClient(mod.create_app())
    headers = {"X-API-Key": "training-demo-key"}
    question = {"question": "What are the clinic opening hours?"}
    assert client.post("/chat", json=question, headers=headers).status_code == 502
    assert client.post("/chat", json=question, headers=headers).status_code == 502
    assert client.post("/chat/stream", json=question, headers=headers).status_code == 502
    metrics = client.get("/metrics", headers=headers).json()
    assert metrics["errors"] >= 3
    assert metrics["cache_hits"] == 0  # failures must never be cached


def test_llm_chat_retries_transient_503_and_sets_timeout(monkeypatch):
    import time as _time

    import litellm

    from agentic_ai_training.llm import chat

    calls = []

    class _Msg:
        content = "hello"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def fake_completion(model, messages, api_key, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise Exception("503 Service Unavailable: model overloaded")
        return _Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(_time, "sleep", lambda s: None)  # skip the 20 s retry wait
    assert chat("sys", "hi") == "hello"
    assert calls == [60, 60]  # every attempt carries the timeout


def test_llm_chat_raises_on_non_transient_error(monkeypatch):
    import litellm

    from agentic_ai_training.llm import chat

    def fake_completion(model, messages, api_key, timeout=None):
        raise ValueError("invalid API key")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    with pytest.raises(ValueError, match="invalid API key"):
        chat("sys", "hi")


def test_live_test_answer_checking():
    # The old whole-stdout substring check was vacuous: "5" matched the
    # echoed question ("What is 18 * 7 + 5?") even with a wrong answer.
    import subprocess

    spec = importlib.util.spec_from_file_location("live_test", ROOT / "scripts" / "live_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT / "labs" / "day1_basic_agent",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert len(mod.extract_answers(proc.stdout)) == 3
    assert mod.check_answers(proc.stdout, ["131", "5", "Paris"]) == []
    assert mod.check_answers(
        "Q: What is 18 * 7 + 5?\nA: I don't know.\n", ["131", "5", "Paris"]
    ) == ["131", "5", "Paris"]
    assert mod.check_answers("A: The answer is 15.\n", ["5"]) == ["5"]
    assert mod.check_answers("A: The answer is 5.\n", ["5"]) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
