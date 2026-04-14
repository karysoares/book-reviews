from langchain_chroma import Chroma

from book_rag.embeddings import build_huggingface_embeddings
from book_rag.settings import Settings


def get_vector_store(settings: Settings) -> Chroma:
    embeddings = build_huggingface_embeddings(settings.embedding_model)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_directory),
    )


def chroma_document_count(store: Chroma) -> int:
    coll = getattr(store, "_collection", None)
    if coll is None:
        return 0
    try:
        return int(coll.count())
    except Exception:
        return 0
