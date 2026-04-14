SYSTEM_INSTRUCTIONS = (
    "You answer using only the provided context. If the context does not "
    "contain enough information, say you do not know. Respond in the same "
    "language as the user's question."
)


def build_user_message(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}"
