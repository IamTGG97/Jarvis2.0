"""Builds the Ollama chat messages that combine the system prompt, retrieved
conversation memory, what the camera saw (if anything), and the user's new
question.

Retrieved turns are placed inside the system message as clearly delimited
"conversation notes" rather than replayed as fake chat messages. On the memory
eval this made qwen2.5:1.5b more willing to answer from the notes and more
likely to admit when a fact was never mentioned (see
docs/overnight/memory-engineer.md for the numbers).

Vision context goes in the same system message inside a ``<vision>`` block.
When there is none, the block says so explicitly: without that line qwen2.5
invented what it "saw" ("I can see your computer and your phone").
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

# Wording measured on qwen2.5:1.5b (docs/overnight/vision-reasoning-backend.md):
# a softer version still produced "Sure, let's check the desk" and "You don't
# look tired"; this one answered "I can't see anything right now" for those
# while still answering from notes ("What color is my car?") and general
# knowledge ("What color is a banana?").
NO_VISION = (
    "IMPORTANT: You have no eyes. No camera image and no object detections were "
    "given to you for this turn, so you cannot see the user, their room, or "
    "anything around them. If a question needs sight (what is on, behind, near "
    "or around the user, how they or something looks, colors, objects, text), "
    "answer: \"I can't see anything right now.\" Do not pretend to look, and "
    "never describe or guess what might be there."
)


def format_memory_notes(conversation_context: list[dict[str, str]]) -> str:
    lines = []
    for turn in conversation_context:
        lines.append(f"User: {turn['transcript']}")
        lines.append(f"Jarvis: {turn['response']}")
    return "\n".join(lines) if lines else NO_NOTES


def build_system_prompt(
    conversation_context: list[dict[str, str]],
    vision_context: str | None = None,
) -> str:
    vision = (vision_context or "").strip() or NO_VISION
    return (
        f"{SYSTEM_PROMPT_INTRO}\n\n"
        "Conversation notes from earlier with this user, oldest first. They are "
        "everything you know about the user:\n"
        f"<notes>\n{format_memory_notes(conversation_context)}\n</notes>\n\n"
        f"{MEMORY_RULES}\n\n"
        f"What the camera shows right now:\n<vision>\n{vision}\n</vision>"
    )


def build_chat_messages(
    transcript: str,
    conversation_context: list[dict[str, str]],
    vision_context: str | None = None,
) -> list[dict[str, str]]:
    """``vision_context`` is a short plain-text summary of this turn's camera
    frame (``reasoning.vision_summary``). None means no vision data, and the
    prompt then tells the model it cannot see."""
    return [
        {"role": "system", "content": build_system_prompt(conversation_context, vision_context)},
        {"role": "user", "content": transcript},
    ]
