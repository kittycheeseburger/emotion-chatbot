import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.ml.sentiment_service import SentimentService
from app.schemas.chat import ChatRequest, ChatResponse, InsightRequest, InsightResponse
from app.services.glm_client import GLMClient
from app.services.insight_service import InsightService
from app.services.openai_client import OpenAIClient

router = APIRouter()
sentiment_service = SentimentService()
glm_client = GLMClient()
openai_client = OpenAIClient()
insight_service = InsightService()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    latest_user_message = request.message.strip()

    if not latest_user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    emotion = sentiment_service.analyze(latest_user_message)

    insight = insight_service.build(
        message=latest_user_message,
        emotion=emotion,
        history=request.history,
    )
    reply = await _build_chat_reply(latest_user_message, request.history, emotion, insight)

    return ChatResponse(reply=reply, emotion=emotion, insight=insight, context_size=len(request.history) + 1)


async def _build_chat_reply(latest_user_message, history, emotion, insight):
    mode = settings.chat_reply_mode.lower().strip()

    if mode == "rag":
        return insight_service.build_reply(latest_user_message, emotion, insight)

    if mode in {"openai", "chatgpt"}:
        try:
            return await openai_client.chat(
                message=latest_user_message,
                history=history,
                emotion=emotion,
                insight=insight,
            )
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500] if error.response is not None else str(error)
            raise HTTPException(status_code=502, detail=f"OpenAI request failed: {detail}") from error
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"OpenAI network error: {error}") from error
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    if mode == "auto":
        try:
            return await openai_client.chat(
                message=latest_user_message,
                history=history,
                emotion=emotion,
                insight=insight,
            )
        except (httpx.HTTPError, RuntimeError):
            return insight_service.build_reply(latest_user_message, emotion, insight)

    try:
        return await glm_client.chat(
            message=latest_user_message,
            history=history,
            emotion=emotion,
        )
    except httpx.HTTPStatusError as error:
        if mode == "auto":
            return insight_service.build_reply(latest_user_message, emotion, insight)
        detail = error.response.text[:500] if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"GLM request failed: {detail}") from error
    except httpx.HTTPError as error:
        if mode == "auto":
            return insight_service.build_reply(latest_user_message, emotion, insight)
        raise HTTPException(status_code=502, detail=f"GLM network error: {error}") from error
    except RuntimeError as error:
        if mode == "auto":
            return insight_service.build_reply(latest_user_message, emotion, insight)
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.post("/insight", response_model=InsightResponse)
async def insight(request: InsightRequest) -> InsightResponse:
    latest_user_message = request.message.strip()

    if not latest_user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    emotion = request.emotion or sentiment_service.analyze(latest_user_message)

    return InsightResponse(
        insight=insight_service.build(
            message=latest_user_message,
            emotion=emotion,
            history=request.history,
            review_mode=request.include_ai_review,
        )
    )
