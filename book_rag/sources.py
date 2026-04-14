from __future__ import annotations

from langchain_core.documents import Document

from book_rag.types import Source

_METADATA_TITLE_KEYS = ("Title", "title")
_SNIPPET_MAX = 180


def _pick_title(meta: dict) -> str | None:
    for k in _METADATA_TITLE_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _format_authors(meta: dict) -> str | None:
    raw = meta.get("authors")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        if parts:
            return ", ".join(parts[:5])
    return None


def _format_categories(meta: dict) -> str | None:
    raw = meta.get("categories")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        if parts:
            return ", ".join(parts[:3])
    return None


def document_to_source(doc: Document) -> Source:
    meta = doc.metadata or {}
    title = _pick_title(meta)
    authors = _format_authors(meta)
    categories = _format_categories(meta)
    bits = []
    if authors:
        bits.append(authors)
    if categories:
        bits.append(categories)
    summary = " · ".join(bits) if bits else None
    if summary is None and doc.page_content:
        text = doc.page_content.strip().replace("\n", " ")
        if len(text) > _SNIPPET_MAX:
            text = text[: _SNIPPET_MAX] + "…"
        summary = text
    return Source(title=title, summary=summary)
