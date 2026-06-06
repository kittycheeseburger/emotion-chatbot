import httpx

from app.core.config import settings
from app.schemas.chat import ChatMessage, EmotionResult


class GLMClient:
    def __init__(self) -> None:
        self.api_key = settings.glm_api_key
        self.base_url = settings.glm_api_base_url.rstrip("/")
        self.model = settings.glm_model

    async def chat(
        self,
        message: str,
        history: list[ChatMessage],
        emotion: EmotionResult,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GLM_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "messages": self._build_messages(message, history, emotion),
            "temperature": 0.7,
            "top_p": 0.9,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError("GLM returned no choices.")

        content = choices[0].get("message", {}).get("content", "")
        return content.strip()

    async def emotion_review(
        self,
        message: str,
        history: list[ChatMessage],
        emotion: EmotionResult,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GLM_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "messages": self._build_review_messages(message, history, emotion),
            "temperature": 0.35,
            "top_p": 0.8,
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError("GLM returned no choices.")

        content = choices[0].get("message", {}).get("content", "")
        return content.strip()

    def _build_messages(
        self,
        message: str,
        history: list[ChatMessage],
        emotion: EmotionResult,
    ) -> list[dict[str, str]]:
        recent_history = history[-12:]
        system_prompt = (
            "你是一个情绪支持型中文聊天机器人。你需要自然地延续上下文，"
            "先回应用户具体内容，再结合情绪状态给出简短、可执行、积极的建议。"
            "语气要温和、尊重、稳定，避免调侃、说教、夸大承诺，避免使用“摆烂”等弱化用户困扰的表达。"
            "建议优先给出一到两条具体下一步。"
            "不要声称自己是医生或心理治疗师，不要做诊断。"
            f"用户最新情绪：{emotion.emoji} {emotion.label}，情绪指数 {emotion.score}/100，"
            f"置信度 {emotion.confidence:.2f}。"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": item.role, "content": item.content} for item in recent_history)
        messages.append({"role": "user", "content": message})
        return messages

    def _build_review_messages(
        self,
        message: str,
        history: list[ChatMessage],
        emotion: EmotionResult,
    ) -> list[dict[str, str]]:
        recent_history = history[-16:]
        system_prompt = (
            "你是一个中文情绪对话复盘助手。请基于聊天上下文和最新情绪，"
            "输出一段 80 字以内的复盘摘要。必须包含：主要情绪、可能触发点、下一步建议。"
            "语气温和具体，不诊断，不夸大，不使用列表。"
        )
        user_prompt = (
            f"最新用户消息：{message}\n"
            f"最新情绪：{emotion.emoji} {emotion.label}，指数 {emotion.score}/100，"
            f"置信度 {emotion.confidence:.2f}。"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": item.role, "content": item.content} for item in recent_history)
        messages.append({"role": "user", "content": user_prompt})
        return messages
