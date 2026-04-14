from book_rag.prompts import SYSTEM_INSTRUCTIONS, build_user_message


def test_system_instructions_non_empty():
    assert len(SYSTEM_INSTRUCTIONS) > 20
    assert "context" in SYSTEM_INSTRUCTIONS.lower()


def test_build_user_message_format():
    msg = build_user_message("ctx block", "What is this?")
    assert "ctx block" in msg
    assert "What is this?" in msg
    assert msg.startswith("Context:")
