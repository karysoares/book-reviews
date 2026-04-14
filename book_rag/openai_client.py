from openai import OpenAI

from book_rag.prompts import SYSTEM_INSTRUCTIONS
from book_rag.settings import Settings


def complete_chat(
    client: OpenAI,
    settings: Settings,
    user_content: str,
) -> str:
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=settings.max_completion_tokens,
    )
    choice = response.choices[0].message
    if choice.content is None:
        return ""
    return choice.content.strip()
