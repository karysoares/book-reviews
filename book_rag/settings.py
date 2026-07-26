import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


class Settings(BaseModel):
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    embedding_model: str = "intfloat/multilingual-e5-large"
    chroma_collection: str = "books"
    chroma_persist_directory: Path = Field(default=Path("books_vector.db"))
    rag_top_k: int = Field(default=5, ge=1, le=50)
    max_completion_tokens: int = Field(default=300, ge=1, le=4096)

    @field_validator("chroma_persist_directory", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @classmethod
    def from_env(cls) -> "Settings":
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        default_model = "gpt-4o-mini"

        # Prefer OpenAI when its key is set; otherwise fall back to Gemini via
        # its OpenAI-compatible endpoint so the same client code path is reused.
        if openai_key:
            key = openai_key
        elif gemini_key:
            key = gemini_key
            if base_url is None:
                base_url = GEMINI_OPENAI_BASE_URL
            default_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        else:
            msg = "OPENAI_API_KEY or GEMINI_API_KEY must be set (both are empty)."
            raise ValueError(msg)

        return cls(
            openai_api_key=key,
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", default_model),
            openai_base_url=base_url,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "intfloat/multilingual-e5-large",
            ),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "books"),
            chroma_persist_directory=Path(
                os.getenv("CHROMA_PERSIST_DIRECTORY", "books_vector.db"),
            ),
            rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
            max_completion_tokens=int(os.getenv("MAX_COMPLETION_TOKENS", "300")),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
