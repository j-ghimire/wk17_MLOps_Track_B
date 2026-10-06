import logging
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .assistant import answer_message
from .config import get_settings
from .schemas import ChatRequest, ChatResponse, Source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title=settings.app_name, version="1.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please try again shortly."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    request_id = str(uuid4())
    started = monotonic()
    result = answer_message(payload.message, settings)
    sources = [Source(title=article.title, content=article.content, score=score) for article, score in result.sources] if payload.include_sources else []
    logger.info(
        "request_id=%s category=%s degraded=%s iterations=%s latency_ms=%.1f",
        request_id,
        result.category,
        result.degraded,
        len(result.steps),
        (monotonic() - started) * 1000,
    )
    return ChatResponse(
        answer=result.answer,
        category=result.category,
        intent=result.intent,
        confidence=result.confidence,
        degraded=result.degraded,
        sources=sources,
        request_id=request_id,
    )
