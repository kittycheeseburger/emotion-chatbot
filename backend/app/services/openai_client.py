import httpx

from app.core.config import settings
from app.schemas.chat import ChatMessage, EmotionInsight, EmotionResult


class OpenAIClient:
    def __init__(self) -> None:
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_api_base_url.rstrip("/")
        self.model = settings.openai_model

    async def chat(
        self,
        message: str,
        history: list[ChatMessage],
        emotion: EmotionResult,
        insight: EmotionInsight,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "instructions": self._build_instructions(emotion, insight),
            "input": self._build_input(message, history),
            "temperature": 0.7,
            "max_output_tokens": 700,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        return self._extract_text(response.json())

    def _build_instructions(self, emotion: EmotionResult, insight: EmotionInsight) -> str:
        return (
            "你是一个情绪支持型中文聊天机器人。请自然延续上下文，先回应用户具体内容，"
            "再结合情绪状态和 RAG 策略给出简短、可执行、积极的建议。"
            "语气要温和、尊重、稳定，避免调侃、说教、夸大承诺，避免弱化用户困扰。"
            "不要声称自己是医生或心理治疗师，不要做诊断。"
            "回复控制在 2 到 4 段，必要时可以给 1 到 2 个具体步骤。"
            f"用户最新情绪：{emotion.emoji} {emotion.label}，情绪指数 {emotion.score}/100，"
            f"置信度 {emotion.confidence:.2f}。"
            f"RAG 画像：触发点={insight.trigger}；策略={insight.strategy}；"
            f"对话目标={insight.goal}；建议语气={insight.tone}；下一步行动={insight.next_action}。"
        )

    def _build_input(self, message: str, history: list[ChatMessage]) -> list[dict[str, str]]:
        recent_history = history[-12:]
        input_items: list[dict[str, str]] = []

        for item in recent_history:
            if item.role == "system":
                continue
            input_items.append({"role": item.role, "content": item.content})

        input_items.append({"role": "user", "content": message})
        return input_items

    def _extract_text(self, data: dict) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: list[str] = []
        for output_item in data.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                text = content_item.get("text")
                if isinstance(text, str):
                    chunks.append(text)

        content = "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
        if not content:
            raise RuntimeError("OpenAI returned no text.")

        return content
