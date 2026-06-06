from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Emotion Chatbot API"
    app_env: str = "development"
    frontend_origin: str = "http://localhost:5173"
    openai_api_key: str = ""
    openai_api_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    glm_api_key: str = ""
    glm_api_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4.7-flash"
    chat_reply_mode: str = "openai"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
