"""Create a minimal books.db and Chroma store for local demos when full CSVs are absent."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from book_rag.embeddings import build_huggingface_embeddings

BOOKS_DB = REPO_ROOT / "books.db"
CHROMA_DIR = REPO_ROOT / "books_vector.db"
MODEL = "intfloat/multilingual-e5-large"


def _row(
    title: str,
    description: str,
    authors: str,
    categories: str,
    publisher: str = "Demo Press",
    year: str = "2020",
) -> dict:
    return {
        "Title": title,
        "description": description,
        "authors": authors,
        "categories": categories,
        "publisher": publisher,
        "publishedDate": year,
        "previewLink": "",
        "image": "",
        "infoLink": "",
        "ratingsCount": "50",
        "review_mean": "4.2",
        "review_ids": "[]",
    }


def demo_catalog() -> pd.DataFrame:
    rows = [
        _row(
            "The Dog Owner's Handbook",
            "Guia prático para escolher raça, alimentação, passeios e treino com reforço "
            "positivo; inclui sinais de alerta veterinário.",
            "['Alex Morgan']",
            "['Pets']",
        ),
        _row(
            "Cats in the Home",
            "Enriquecimento ambiental, caixas de areia, nutrição e quando procurar o "
            "veterinário para problemas comuns em gatos de interior.",
            "['Sam Rivera']",
            "['Pets']",
        ),
        _row(
            "História do Litoral Português",
            "Rotas comerciais, comunidades piscatórias e fortificações ao longo do "
            "Atlântico desde o século XV.",
            "['Maria Silva']",
            "['History']",
        ),
        _row(
            "Python for Data Work",
            "Introdução a pandas, limpeza de dados e pipelines reprodutíveis com testes "
            "leves e empacotamento de projetos.",
            "['Jordan Lee']",
            "['Computers']",
        ),
    ]
    return pd.DataFrame(rows)


def compact_metadata(row: pd.Series) -> dict:
    meta = {}
    for key in (
        "Title",
        "authors",
        "categories",
        "publisher",
        "publishedDate",
        "previewLink",
    ):
        val = row.get(key, "")
        if val is None or (isinstance(val, str) and not str(val).strip()):
            continue
        meta[key] = val
    return meta


def main() -> None:
    df = demo_catalog()
    engine = create_engine(f"sqlite:///{BOOKS_DB}")
    df.to_sql("books_data", engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows to {BOOKS_DB}")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True)

    embeddings = build_huggingface_embeddings(MODEL)
    vector_store = Chroma(
        collection_name="books",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    documents: list[Document] = []
    for _, row in df.iterrows():
        meta = compact_metadata(row)
        desc = row.get("description", "")
        title = row.get("Title", "")
        if isinstance(desc, str) and desc.strip():
            documents.append(Document(page_content=desc, metadata=meta))
        if isinstance(title, str) and title.strip():
            documents.append(Document(page_content=title, metadata=meta))

    batch = 32
    for i in range(0, len(documents), batch):
        part = documents[i : i + batch]
        vector_store.add_documents(part)
        print(f"Indexed {min(i + batch, len(documents))} / {len(documents)}")

    coll = getattr(vector_store, "_collection", None)
    n = int(coll.count()) if coll is not None else -1
    print(f"Chroma vectors: {n}")


if __name__ == "__main__":
    main()
