import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
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
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            msg = "OPENAI_API_KEY is not set or is empty."
            raise ValueError(msg)
        return cls(
            openai_api_key=key,
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
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
