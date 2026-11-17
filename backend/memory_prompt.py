"""Builds the Ollama chat messages that combine the system prompt, retrieved
conversation memory, and the user's new question.

Retrieved turns are placed inside the system message as clearly delimited
"conversation notes" rather than replayed as fake chat messages. On the memory
eval this made qwen2.5:1.5b more willing to answer from the notes and more
likely to admit when a fact was never mentioned (see
docs/overnight/memory-engineer.md for the numbers).
"""

SYSTEM_PROMPT_INTRO = "You are Jarvis, a concise and helpful voice assistant."

MEMORY_RULES = (
    "Rules:\n"
    "- When the user tells you something about themselves, acknowledge it "
    "briefly and naturally; you will remember it.\n"
    "- When the user asks about themselves (their name, family, pets, job, "
    "preferences, plans, or appointments), answer only from the conversation "
    "notes, and say it naturally, for example: 'You told me ...'.\n"
    "- If the notes do not contain the answer, say you haven't been told: "
    "'You haven't mentioned that to me yet.' Never guess or invent personal details.\n"
    "- For general questions, answer normally in one or two short sentences."
)

NO_NOTES = "(no earlier conversation)"


def format_memory_notes(conversation_context: list[dict[str, str]]) -> str:
    lines = []
    for turn in conversation_context:
        lines.append(f"User: {turn['transcript']}")
        lines.append(f"Jarvis: {turn['response']}")
    return "\n".join(lines) if lines else NO_NOTES


def build_system_prompt(conversation_context: list[dict[str, str]]) -> str:
    return (
        f"{SYSTEM_PROMPT_INTRO}\n\n"
        "Conversation notes from earlier with this user, oldest first. They are "
        "everything you know about the user:\n"
        f"<notes>\n{format_memory_notes(conversation_context)}\n</notes>\n\n"
        f"{MEMORY_RULES}"
    )


def build_chat_messages(
    transcript: str,
    conversation_context: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(conversation_context)},
        {"role": "user", "content": transcript},
    ]
