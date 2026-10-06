from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    include_sources: bool = True


class Source(BaseModel):
    title: str
    content: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    category: str
    intent: str
    confidence: float
    degraded: bool
    sources: list[Source] = Field(default_factory=list)
    request_id: str
