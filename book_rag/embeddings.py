from langchain_huggingface import HuggingFaceEmbeddings


class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        t = text.strip()
        if t.lower().startswith("query:"):
            return super().embed_query(t)
        return super().embed_query(f"query: {t}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = []
        for raw in texts:
            t = raw.strip()
            low = t.lower()
            if low.startswith("passage:") or low.startswith("query:"):
                prefixed.append(t)
            else:
                prefixed.append(f"passage: {t}")
        return super().embed_documents(prefixed)


def build_huggingface_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    if "e5" in model_name.lower():
        return E5HuggingFaceEmbeddings(model_name=model_name)
    return HuggingFaceEmbeddings(model_name=model_name)
