from __future__ import annotations

from typing import Any, Protocol

from openai import APIError, OpenAI, OpenAIError

from book_rag.openai_client import complete_chat
from book_rag.prompts import build_user_message
from book_rag.settings import Settings
from book_rag.sources import document_to_source
from book_rag.types import QAResult


class SupportsSimilaritySearch(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Any]:
        ...


class QAService:
    def __init__(
        self,
        settings: Settings,
        vector_store: SupportsSimilaritySearch,
        openai_client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._client = openai_client or OpenAI(api_key=settings.openai_api_key)

    def ask(self, question: str) -> QAResult:
        q = question.strip()
        if not q:
            return QAResult(answer="Please enter a non-empty question.", sources=[])

        docs = self._vector_store.similarity_search(
            q,
            k=self._settings.rag_top_k,
        )
        if not docs:
            return QAResult(
                answer=(
                    "No relevant passages were retrieved from the index. "
                    "Try rephrasing or check that the vector store is built."
                ),
                sources=[],
            )

        context = "\n\n".join(d.page_content for d in docs)
        user_msg = build_user_message(context, q)
        sources = [document_to_source(d) for d in docs]

        try:
            answer = complete_chat(self._client, self._settings, user_msg)
        except (APIError, OpenAIError) as exc:
            return QAResult(
                answer=f"The language model request failed: {exc}",
                sources=sources,
            )

        return QAResult(answer=answer or "(Empty response.)", sources=sources)


def answer_question(
    question: str,
    settings: Settings,
    vector_store: SupportsSimilaritySearch,
    openai_client: OpenAI | None = None,
) -> QAResult:
    return QAService(settings, vector_store, openai_client).ask(question)
