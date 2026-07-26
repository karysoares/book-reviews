from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="Book QA", layout="centered")
st.title("Book QA System")

try:
    from book_rag.qa_service import QAService
    from book_rag.settings import get_settings
    from book_rag.vector_store import chroma_document_count, get_vector_store
except ImportError:
    st.error("Run from the repository root so the book_rag package can be imported.")
    st.stop()


@st.cache_resource
def _boot():
    settings = get_settings()
    store = get_vector_store(settings)
    n_docs = chroma_document_count(store)
    return settings, store, n_docs


try:
    settings, vector_store, index_size = _boot()
except ValueError as exc:
    st.error(str(exc))
    st.info(
        "Create a `.env` file next to `rag_pipeline.py` with either:\n\n"
        "`OPENAI_API_KEY=sk-...`\n\n"
        "or, to use Google Gemini:\n\n"
        "`GEMINI_API_KEY=...`\n\n"
        f"Expected path: `{REPO_ROOT / '.env'}`"
    )
    st.stop()

if index_size == 0:
    st.error("The Chroma index has no vectors yet.")
    st.info(
        f"Expected store path: `{settings.chroma_persist_directory.resolve()}`. "
        "Run `notebooks/05_indexing_pipeline.ipynb` after `books.db` exists, "
        "or point `CHROMA_PERSIST_DIRECTORY` to a folder that already contains "
        "an indexed `books` collection. "
        "If you indexed **before** E5 `query:`/`passage:` prefixes were added, "
        "re-run the indexing notebook once so embeddings match the app."
    )
    st.stop()

st.success(f"Models and index are ready ({index_size} vectors).")
st.header("Ask me anything about books")
question = st.text_input("Your question:")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer…"):
            result = QAService(settings, vector_store).ask(question)
        st.markdown("**Answer**")
        st.write(result.answer)
        if result.sources:
            st.markdown("**Sources**")
            for i, src in enumerate(result.sources, start=1):
                line = src.title or "Untitled"
                if src.summary:
                    line = f"{line} — {src.summary}"
                st.caption(f"{i}. {line}")
