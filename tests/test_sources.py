from langchain_core.documents import Document

from book_rag.sources import document_to_source


def test_document_to_source_prefers_title_and_authors():
    doc = Document(
        page_content="Long description " * 50,
        metadata={"Title": "My Book", "authors": ["A. Writer", "B. Co"]},
    )
    src = document_to_source(doc)
    assert src.title == "My Book"
    assert src.summary is not None
    assert "A. Writer" in src.summary
