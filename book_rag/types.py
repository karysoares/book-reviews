from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str | None = None
    summary: str | None = Field(
        default=None,
        description="Short line from metadata or truncated chunk text",
    )


class QAResult(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
