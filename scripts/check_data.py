"""Print what is present for the full pipeline (data/ CSVs, books.db, Chroma)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CSV_PAIR = ("books_data.csv", "books_rating.csv")


def main() -> None:
    print(f"Repo: {REPO}\n")
    missing_csv = [n for n in CSV_PAIR if not (DATA / n).exists()]
    if missing_csv:
        print("Faltam CSVs em data/:")
        for n in missing_csv:
            print(f"  - {DATA / n}")
        print(
            "\nColoca aqui os ficheiros do teu backup (ou da mesma fonte original do projeto). "
            "Depois corre notebooks/01_data_preparation.ipynb e notebooks/05_indexing_pipeline.ipynb."
        )
    else:
        print("CSVs encontrados:")
        for n in CSV_PAIR:
            p = DATA / n
            print(f"  OK {p} ({p.stat().st_size // 1_000_000} MB approx)")

    db = REPO / "books.db"
    print(f"\nbooks.db: {'OK' if db.exists() else 'ausente — corre o01 após os CSVs'}")

    chroma = REPO / "books_vector.db"
    if chroma.exists():
        sqlite = chroma / "chroma.sqlite3"
        print(f"books_vector.db: pasta existe ({'com sqlite' if sqlite.exists() else 'vazia?'})")
    else:
        print("books_vector.db: ausente — corre o 05 após books.db")

    return 1 if missing_csv or not db.exists() else 0


if __name__ == "__main__":
    sys.exit(main())
