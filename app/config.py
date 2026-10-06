from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Part B Agentic MLOps Support Assistant"
    llm_provider: str = "none"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    request_timeout_seconds: float = 20.0
    max_message_chars: int = 2000
    max_iterations: int = Field(default=3, ge=1, le=8)
    retrieval_top_k: int = Field(default=3, ge=1, le=10)
    min_evidence_score: float = Field(default=0.18, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
