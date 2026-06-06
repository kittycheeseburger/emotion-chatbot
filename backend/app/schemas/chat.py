from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class EmotionResult(BaseModel):
    label: str
    emoji: str
    score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)


class EmotionInsight(BaseModel):
    trigger: str
    goal: str
    next_action: str
    tone: str
    summary: str
    strategy: str = ""
    retrieval_score: float = Field(default=0, ge=0, le=1)
    source: str = "local"


class ChatResponse(BaseModel):
    reply: str
    emotion: EmotionResult
    insight: EmotionInsight
    context_size: int


class InsightRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    emotion: EmotionResult | None = None
    include_ai_review: bool = False


class InsightResponse(BaseModel):
    insight: EmotionInsight
