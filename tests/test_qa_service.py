from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from openai import APIError

from book_rag.qa_service import QAService
from book_rag.settings import Settings


def _settings():
    return Settings(openai_api_key="sk-test", rag_top_k=2)


def test_ask_empty_question():
    store = MagicMock()
    svc = QAService(_settings(), store, openai_client=MagicMock())
    out = svc.ask("   ")
    assert "empty" in out.answer.lower() or "Please enter" in out.answer
    store.similarity_search.assert_not_called()


def test_ask_no_retrieval():
    store = MagicMock()
    store.similarity_search.return_value = []
    client = MagicMock()
    svc = QAService(_settings(), store, openai_client=client)
    out = svc.ask("hello")
    assert "No relevant" in out.answer or "retrieved" in out.answer
    client.chat.completions.create.assert_not_called()


def test_ask_success():
    docs = [
        Document(page_content="Plot summary.", metadata={"Title": "T1"}),
        Document(page_content="More text.", metadata={"Title": "T2"}),
    ]
    store = MagicMock()
    store.similarity_search.return_value = docs

    client = MagicMock()
    msg = MagicMock()
    msg.content = "Synthetic answer."
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]

    svc = QAService(_settings(), store, openai_client=client)
    out = svc.ask("What happens?")

    assert out.answer == "Synthetic answer."
    assert len(out.sources) == 2
    call_kw = client.chat.completions.create.call_args.kwargs
    user = call_kw["messages"][1]["content"]
    assert "Plot summary." in user
    assert "What happens?" in user


def test_ask_openai_error_surfaces_message():
    store = MagicMock()
    store.similarity_search.return_value = [
        Document(page_content="x", metadata={}),
    ]
    client = MagicMock()
    client.chat.completions.create.side_effect = APIError(
        message="fail",
        request=MagicMock(),
        body=None,
    )
    svc = QAService(_settings(), store, openai_client=client)
    out = svc.ask("q")
    assert "failed" in out.answer.lower()
    assert len(out.sources) == 1
