from collections.abc import AsyncIterator
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.schemas import Source, Turn

SYSTEM_PROMPT = """You are Nugget, a precise research assistant that answers questions from the user's own documents.

Rules:
- Ground every claim in the numbered context passages. Cite them inline like [1] or [2][3], right after the claim they support.
- If the passages do not contain the answer, say so plainly and briefly suggest what document might help. Never invent facts or citations.
- Always reply in the same language as the user's latest question, even when the passages are in another language. If the question's language is ambiguous, use {locale_name}.
- Use Markdown when it improves readability: short paragraphs, bullet lists, tables for comparisons, `code` for identifiers.
- Be direct. Lead with the answer, then supporting detail. No preamble such as "Based on the context"."""

REWRITE_PROMPT = """Rewrite the user's latest message into a standalone search query that can be understood without the conversation. Keep the original language, keep names and key terms, drop filler. Output only the query.

Conversation:
{history}

Latest message: {question}"""

LOCALE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ar": "Arabic",
    "ja": "Japanese",
    "it": "Italian",
}


class LLMNotConfiguredError(RuntimeError):
    pass


@lru_cache
def chat_model() -> BaseChatModel:
    settings = get_settings()
    if not settings.api_key:
        raise LLMNotConfiguredError("NUGGET_API_KEY is not set")
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.api_base,
        temperature=settings.temperature,
        timeout=120,
        max_retries=2,
    )


def format_context(sources: list[Source]) -> str:
    if not sources:
        return "(no passages matched)"
    blocks = []
    for i, s in enumerate(sources, 1):
        where = f"{s.source}, p. {s.page}" if s.page else s.source
        blocks.append(f"[{i}] ({where})\n{s.content}")
    return "\n\n".join(blocks)


def build_messages(
    question: str, history: list[Turn], sources: list[Source], locale: str | None
) -> list[BaseMessage]:
    locale_name = LOCALE_NAMES.get((locale or "en").split("-")[0], "English")
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT.format(locale_name=locale_name))]
    for turn in history[-6:]:
        cls = HumanMessage if turn.role == "user" else AIMessage
        messages.append(cls(turn.content))
    messages.append(
        HumanMessage(f"Context passages:\n\n{format_context(sources)}\n\n---\nQuestion: {question}")
    )
    return messages


async def rewrite_query(question: str, history: list[Turn]) -> str:
    """Turn a follow-up like "and in 2023?" into a query retrieval can actually use."""
    if not history:
        return question
    transcript = "\n".join(f"{t.role}: {t.content[:600]}" for t in history[-4:])
    try:
        result = await chat_model().ainvoke(
            REWRITE_PROMPT.format(history=transcript, question=question)
        )
    except Exception:
        return question
    text = str(result.content).strip().strip('"').strip()
    return text if 0 < len(text) < 500 else question


async def stream_answer(messages: list[BaseMessage]) -> AsyncIterator[str]:
    async for chunk in chat_model().astream(messages):
        if chunk.content:
            yield str(chunk.content)
