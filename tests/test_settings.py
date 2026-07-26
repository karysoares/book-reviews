import pytest

from book_rag.settings import Settings


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_from_env_loads_optional_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    s = Settings.from_env()
    assert s.openai_api_key == "sk-test"
    assert s.rag_top_k == 3
    assert s.openai_chat_model == "gpt-4o-mini"
    assert s.openai_base_url is None


def test_from_env_uses_gemini_when_only_gemini_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    s = Settings.from_env()
    assert s.openai_api_key == "gemini-test"
    assert s.openai_base_url and "generativelanguage" in s.openai_base_url
    assert s.openai_chat_model == "gemini-flash-latest"


def test_settings_manual_construct():
    s = Settings(openai_api_key="k")
    assert s.chroma_collection == "books"
    assert s.rag_top_k == 5
