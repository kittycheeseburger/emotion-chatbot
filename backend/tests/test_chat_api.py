from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.schemas.chat import EmotionResult


class FakeGLMClient:
    async def chat(self, message, history, emotion):
        return f"收到：{message}。当前情绪是 {emotion.label}，可以先做一个小步骤。"


class FakeOpenAIClient:
    async def chat(self, message, history, emotion, insight):
        return f"ChatGPT 回复：{message}。我会结合{emotion.label}和{insight.trigger}继续聊。"


class FailingOpenAIClient:
    async def chat(self, message, history, emotion, insight):
        raise RuntimeError("openai unavailable")


class FailingGLMClient:
    async def chat(self, message, history, emotion):
        raise RuntimeError("busy")


class FakeSentimentService:
    def analyze(self, text):
        return EmotionResult(label="焦虑", emoji="😟", score=42, confidence=0.76)


def test_chat_returns_reply_and_emotion(monkeypatch):
    monkeypatch.setattr(routes, "glm_client", FakeGLMClient())
    monkeypatch.setattr(routes, "openai_client", FakeOpenAIClient())
    monkeypatch.setattr(routes, "sentiment_service", FakeSentimentService())
    monkeypatch.setattr(routes.settings, "chat_reply_mode", "openai")

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "我今天有点焦虑，工作卡住了",
            "history": [],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"].startswith("ChatGPT 回复")
    assert data["emotion"]["label"] == "焦虑"
    assert data["emotion"]["emoji"] == "😟"
    assert data["insight"]["trigger"] == "工作压力"
    assert data["insight"]["source"] == "rag"
    assert data["insight"]["strategy"]
    assert data["insight"]["retrieval_score"] > 0
    assert data["context_size"] == 1


def test_chat_auto_mode_falls_back_to_rag_when_openai_fails(monkeypatch):
    monkeypatch.setattr(routes, "openai_client", FailingOpenAIClient())
    monkeypatch.setattr(routes, "glm_client", FailingGLMClient())
    monkeypatch.setattr(routes, "sentiment_service", FakeSentimentService())
    monkeypatch.setattr(routes.settings, "chat_reply_mode", "auto")

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "我今天有点焦虑，工作卡住了",
            "history": [],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "工作压力" in data["reply"]
    assert data["insight"]["source"] == "rag"


def test_chat_rejects_empty_message():
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "   ", "history": []})

    assert response.status_code == 400


def test_insight_returns_rag_review_when_requested(monkeypatch):
    monkeypatch.setattr(routes, "glm_client", FakeGLMClient())
    monkeypatch.setattr(routes, "sentiment_service", FakeSentimentService())

    client = TestClient(app)
    response = client.post(
        "/api/insight",
        json={
            "message": "我今天工作卡住了，很焦虑",
            "history": [],
            "emotion": {"label": "焦虑", "emoji": "😟", "score": 42, "confidence": 0.76},
            "include_ai_review": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["insight"]["source"] == "rag"
    assert data["insight"]["trigger"] == "工作压力"
    assert data["insight"]["strategy"] == "工作卡住拆解"
    assert "RAG 检索" in data["insight"]["summary"]


def test_insight_retrieves_academic_setback_strategy(monkeypatch):
    monkeypatch.setattr(routes, "glm_client", FakeGLMClient())
    monkeypatch.setattr(routes, "sentiment_service", FakeSentimentService())

    client = TestClient(app)
    response = client.post(
        "/api/insight",
        json={
            "message": "我今天考砸了，真的很难过",
            "history": [],
            "emotion": {"label": "低落", "emoji": "😔", "score": 28, "confidence": 0.9},
            "include_ai_review": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["insight"]["source"] == "rag"
    assert data["insight"]["trigger"] == "学业受挫"
    assert data["insight"]["strategy"] == "学业受挫复盘"
