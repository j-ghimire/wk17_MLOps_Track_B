from fastapi.testclient import TestClient

from app.assistant import AgentConfig, answer_locally
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_returns_grounded_sources():
    response = client.post("/chat", json={"message": "I forgot my password and did not get the reset email"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "ACCOUNT"
    assert body["sources"]
    assert body["request_id"]


def test_empty_prompt_is_clarified():
    result = answer_locally("   ")
    assert result.intent == "ask_for_clarification"
    assert result.steps[0].action == "guardrail"


def test_weak_evidence_requests_clarification():
    config = AgentConfig(retrieval_top_k=3, min_evidence_score=0.9, max_iterations=1)
    result = answer_locally("I forgot my password and need a reset", config)
    assert result.intent == "ask_for_clarification"
    assert result.steps[-1].action == "ask_user"


def test_agent_iteration_limit_is_respected():
    config = AgentConfig(retrieval_top_k=3, min_evidence_score=0.0, max_iterations=1)
    result = answer_locally("card payment and account email are both failing", config)
    retrieval_steps = [step for step in result.steps if step.action in {"retrieve", "search_again"}]
    assert len(retrieval_steps) <= 1


def test_chat_rejects_empty_payload():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
