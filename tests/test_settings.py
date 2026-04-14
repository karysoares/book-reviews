import pytest

from book_rag.settings import Settings


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_from_env_loads_optional_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    s = Settings.from_env()
    assert s.openai_api_key == "sk-test"
    assert s.rag_top_k == 3
    assert s.openai_chat_model == "gpt-4o-mini"


def test_settings_manual_construct():
    s = Settings(openai_api_key="k")
    assert s.chroma_collection == "books"
    assert s.rag_top_k == 5
