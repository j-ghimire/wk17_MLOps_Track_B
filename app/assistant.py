import re
from dataclasses import dataclass, field

from .config import Settings
from .knowledge import Article, search_articles


@dataclass(frozen=True)
class AgentStep:
    action: str
    query: str | None = None
    summary: str | None = None
    evidence: list[str] = field(default_factory=list)
    should_stop: bool = False


@dataclass(frozen=True)
class RoutingResult:
    category: str
    intent: str
    confidence: float
    answer: str
    sources: list[tuple[Article, float]]
    degraded: bool
    steps: list[AgentStep] = field(default_factory=list)


@dataclass(frozen=True)
class AgentConfig:
    retrieval_top_k: int = 3
    min_evidence_score: float = 0.18
    max_iterations: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.retrieval_top_k <= 10:
            raise ValueError("retrieval_top_k must be between 1 and 10")
        if not 0.0 <= self.min_evidence_score <= 1.0:
            raise ValueError("min_evidence_score must be between 0 and 1")
        if not 1 <= self.max_iterations <= 8:
            raise ValueError("max_iterations must be between 1 and 8")


_KEYWORDS = {
    "ACCOUNT": {"password", "login", "account", "sign in", "email"},
    "CARD": {"card", "stolen", "lost", "declined", "freeze"},
    "CASH_WITHDRAWAL": {"cash", "withdraw", "atm", "pin", "fee"},
    "CASH_DEPOSIT": {"deposit", "balance", "receipt"},
    "TRANSFER": {"transfer", "recipient", "bank", "duplicate"},
    "REFUND": {"refund", "returned", "merchant", "money"},
    "EXCHANGE_RATE": {"exchange", "currency", "rate", "fee"},
    "CARD_DELIVERY": {"delivery", "shipping", "track", "address"},
    "TOP_UP": {"top up", "recharge", "add money", "payment"},
}


def _route(message: str) -> tuple[str, float]:
    lowered = message.lower()
    scores = {
        category: sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", lowered) or keyword in lowered)
        for category, keywords in _KEYWORDS.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "GENERAL", 0.25
    return category, min(0.55 + score * 0.12, 0.95)


def _choose_next_action(query: str, matches: list[tuple[Article, float]], initial_category: str) -> tuple[str, str | None]:
    if not matches:
        return "ask_user", "I could not find a directly matching support article. Please tell me whether this is a password, card, transfer, or cash issue so I can narrow it down."
    top_article = matches[0][0]
    article_terms = set(top_article.terms)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    overlap = len(query_terms & article_terms)
    if overlap == 0:
        return "search_again", f"I found a related article, but it does not match the request closely enough. I should search again for {initial_category.lower().replace('_', ' ')}-related guidance."
    if top_article.category != initial_category and initial_category != "GENERAL":
        return "verify_with_secondary_source", f"The first result points to {top_article.category}, so I am verifying whether the user asked about a different issue before finalizing."
    return "answer", None


def _summarize_matches(matches: list[tuple[Article, float]]) -> str:
    if not matches:
        return "No relevant article found."
    chunks = [f"{article.title} ({article.category}, score={score})" for article, score in matches]
    return "; ".join(chunks)


def answer_locally(message: str, config: AgentConfig | None = None) -> RoutingResult:
    config = config or AgentConfig()
    if not message.strip():
        return RoutingResult(
            category="GENERAL",
            intent="ask_for_clarification",
            confidence=0.0,
            answer="Please describe the support issue so I can search the help articles.",
            sources=[],
            degraded=True,
            steps=[AgentStep(action="guardrail", summary="Empty prompt rejected.", should_stop=True)],
        )

    category, confidence = _route(message)
    search_query = message
    all_matches: list[tuple[Article, float]] = []
    steps: list[AgentStep] = []
    max_iterations = max(1, config.max_iterations)

    for iteration in range(max_iterations):
        matches = search_articles(search_query, limit=max(1, config.retrieval_top_k))
        all_matches = sorted({*all_matches, *matches}, key=lambda item: item[1], reverse=True)
        steps.append(
            AgentStep(
                action="retrieve" if iteration == 0 else "search_again",
                query=search_query,
                summary=f"Retrieved {len(matches)} candidate articles.",
                evidence=[_summarize_matches(matches)],
            )
        )

        if not matches:
            if iteration + 1 < max_iterations and category != "GENERAL":
                search_query = f"{category.lower().replace('_', ' ')} {message}"
                continue
            break

        article, score = matches[0]
        action, extra_msg = _choose_next_action(message, matches, category)
        if score < config.min_evidence_score:
            action = "ask_user"
            extra_msg = "I found only weakly related guidance. Please add a detail such as the account, card, transfer, or ATM involved."

        if action == "answer":
            intent = article.title.lower().replace(" ", "_")
            final_confidence = min(0.95, max(confidence, 0.45 + score / 2))
            steps.append(
                AgentStep(
                    action="verify",
                    summary="Evidence passed the relevance threshold and matches the requested issue.",
                    evidence=[article.content],
                    should_stop=True,
                )
            )
            return RoutingResult(
                category=article.category,
                intent=intent,
                confidence=round(final_confidence, 2),
                answer=article.content,
                sources=all_matches[: config.retrieval_top_k],
                degraded=True,
                steps=steps,
            )

        if action in {"search_again", "verify_with_secondary_source"} and iteration + 1 < max_iterations:
            search_query = f"{category.lower().replace('_', ' ')} {message}"
            continue

        if action == "search_again" and matches:
            article, score = matches[0]
            steps.append(AgentStep(action="answer_from_search", summary="Used the best result after the bounded retrieval loop.", evidence=[article.content], should_stop=True))
            return RoutingResult(
                category=article.category,
                intent=article.title.lower().replace(" ", "_"),
                confidence=round(min(0.95, max(confidence, 0.45 + score / 2)), 2),
                answer=article.content,
                sources=all_matches[: config.retrieval_top_k],
                degraded=True,
                steps=steps,
            )

        steps.append(
            AgentStep(
                action="ask_user",
                summary=extra_msg or "Evidence was insufficient; requested more detail instead of guessing.",
                evidence=[_summarize_matches(all_matches)],
                should_stop=True,
            )
        )
        return RoutingResult(
            category="GENERAL" if not all_matches else category,
            intent="ask_for_clarification",
            confidence=0.4,
            answer=extra_msg or "I need one more detail to answer accurately.",
            sources=all_matches[: config.retrieval_top_k],
            degraded=True,
            steps=steps,
        )

    steps.append(
        AgentStep(
            action="ask_user",
            summary="No sufficiently relevant article was found within the configured search limit.",
            evidence=[_summarize_matches(all_matches)],
            should_stop=True,
        )
    )
    return RoutingResult(
        category="GENERAL",
        intent="ask_for_clarification",
        confidence=0.35,
        answer="I could not find a sufficiently relevant support article. Please share a little more detail so I can narrow down the issue.",
        sources=all_matches[: config.retrieval_top_k],
        degraded=True,
        steps=steps,
    )


def answer_message(
    message: str,
    settings: Settings,
    agent_config: AgentConfig | None = None,
) -> RoutingResult:
    if len(message) > settings.max_message_chars:
        return RoutingResult(
            category="GENERAL",
            intent="input_too_long",
            confidence=0.0,
            answer="Your message is too long for this assistant. Please shorten it and try again.",
            sources=[],
            degraded=True,
            steps=[AgentStep(action="guardrail", summary="Prompt length exceeded configured limit.", should_stop=True)],
        )

    result = answer_locally(
        message,
        agent_config
        or AgentConfig(
            retrieval_top_k=settings.retrieval_top_k,
            min_evidence_score=settings.min_evidence_score,
            max_iterations=settings.max_iterations,
        ),
    )
    if settings.llm_provider.lower() not in {"openai", "openai-compatible"} or not settings.openai_api_key:
        return result

    if result.category == "GENERAL" and result.intent == "ask_for_clarification":
        return result

    try:
        from openai import OpenAI
    except Exception:
        return result

    context = "\n".join(article.content for article, _ in result.sources[:3])
    if not context:
        return result

    system_prompt = (
        "You are a careful support assistant. Use only the supplied context, explain uncertainty when evidence is weak, "
        "and never invent a policy or procedure. Return a JSON object with keys: answer, category, intent, confidence."
    )
    try:
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)
        completion = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"},
            ],
        )
        payload = completion.choices[0].message.content
        if not payload:
            return result
        import json

        data = json.loads(payload)
        answer = str(data.get("answer", result.answer))
        category = str(data.get("category", result.category))
        intent = str(data.get("intent", result.intent))
        confidence = float(data.get("confidence", result.confidence))
        if category not in {*_KEYWORDS, "GENERAL"} or not answer.strip():
            raise ValueError("Provider returned an invalid structured response")
        final_result = RoutingResult(
            category=category,
            intent=intent,
            confidence=max(0.0, min(0.99, confidence)),
            answer=answer,
            sources=result.sources,
            degraded=False,
            steps=result.steps,
        )
        final_result.steps.append(AgentStep(action="llm_verify", summary="LLM rechecked evidence and returned a final answer.", should_stop=True))
        return final_result
    except Exception:
        return result
