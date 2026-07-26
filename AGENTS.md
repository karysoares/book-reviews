# AGENTS.md

## Cursor Cloud specific instructions

### Product
Single Python product: a **Book QA / RAG system**. A Streamlit UI (`rag_pipeline.py`) takes a natural-language book question, retrieves passages from a local **Chroma** vector store (`books_vector.db/`) using HuggingFace **E5** embeddings, and asks the **OpenAI** chat API to synthesize a grounded answer with sources. Chroma and SQLite are embedded (file-based), not separate servers. See `README.md` (Portuguese) for the full data pipeline.

### Environment
- Runs on the system **Python 3.12** inside a virtualenv at `.venv/` (the repo README mentions 3.10.12, but 3.12 works for lint, tests, indexing, and the app). Activate with `. .venv/bin/activate` or call binaries directly via `.venv/bin/...`.
- The startup update script (see below) refreshes deps into `.venv/`. `.venv/`, `books.db`, `books_vector.db/`, `.env`, and `data/` are gitignored.

### Lint / test / build-run commands
Standard commands are in `.github/workflows/ci.yml` and `README.md`. Run them from the repo root with the venv active:
- Lint: `flake8 book_rag tests rag_pipeline.py --count --select=E9,F63,F7,F82 --show-source --statistics` (the second `--exit-zero` style pass in CI is non-blocking).
- Tests: `pytest` — 10 tests, fully mock OpenAI and the vector store, so they need **no API key and no index**.
- Run app: `streamlit run rag_pipeline.py` (add `--server.headless true` in cloud).

### Non-obvious gotchas
- **The Streamlit app hard-stops on boot** if `OPENAI_API_KEY` is empty (it only checks non-empty; any value passes boot) OR if the Chroma index has 0 vectors. Set the key via a `.env` file next to `rag_pipeline.py` or an env var.
- **A real `OPENAI_API_KEY` is required only for the final answer generation.** Retrieval (Chroma similarity search) works without it; with an invalid/dummy key the app still boots and shows real Sources, but the Answer shows a graceful "language model request failed" 401 message.
- **Build the vector index before serving.** With no dataset, run `python scripts/build_demo_index.py` to create `books.db` + a small demo `books_vector.db/` (4 fictional books, 8 vectors). It does NOT need an OpenAI key. For the full catalog, place CSVs in `data/` and run `notebooks/01_data_preparation.ipynb` then `notebooks/05_indexing_pipeline.ipynb`.
- **First embedding use downloads `intfloat/multilingual-e5-large` (~GBs) from HuggingFace Hub**, then caches under `~/.cache/huggingface`. The demo build and the app reuse that cache. `HF_TOKEN` is optional (only affects rate limits).
- `scripts/check_data.py` reports which local data artifacts (`books.db`, `books_vector.db/`, CSVs) are present/missing.
