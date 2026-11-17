"""Compare the keyword router with an LLM classifier on labeled questions.

Usage:

    .\\.venv\\Scripts\\python.exe backend\\eval_router.py            # rule router only
    .\\.venv\\Scripts\\python.exe backend\\eval_router.py --llm qwen2.5:0.5b --keep-alive 0

Every question is asked as if a camera frame and detections were attached
(the case where the choice matters). Labels were written by hand before
running. The LLM classifier gets a short instruction and must answer with one
word: text, detections or vlm.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reasoning  # noqa: E402

SCENE = [
    reasoning.Detection("person", 0.9, 0.5, 0.1, 0.45, 0.85),
    reasoning.Detection("laptop", 0.85, 0.3, 0.3, 0.25, 0.3),
    reasoning.Detection("bottle", 0.7, 0.23, 0.46, 0.06, 0.22),
]

# (question, expected route) - held out from the unit tests where possible.
LABELED = [
    ("What's the capital of Japan?", "text"),
    ("How many minutes are in an hour?", "text"),
    ("What did I tell you my sister's name was?", "text"),
    ("Can you recommend a good book?", "text"),
    ("Why is the sky blue?", "text"),
    ("My favorite color is purple.", "text"),
    ("What time is it in London?", "text"),
    ("Write a haiku about coffee.", "text"),
    ("What can you see right now?", "detections"),
    ("How many people are in the room?", "detections"),
    ("Is there anybody behind me?", "detections"),
    ("Do you see my laptop?", "detections"),
    ("Where's the bottle?", "detections"),
    ("Is there a dog in the picture?", "detections"),
    ("Count the chairs you can see.", "detections"),
    ("Am I on camera?", "detections"),
    ("What color is this bottle?", "vlm"),
    ("What does the label on the bottle say?", "vlm"),
    ("What brand is my laptop?", "vlm"),
    ("Is the bottle full or empty?", "vlm"),
    ("Describe what I'm wearing.", "vlm"),
    ("What am I holding up?", "vlm"),
    ("Which is closer to the camera, the bottle or the laptop?", "vlm"),
    ("Can you read this for me?", "vlm"),
    ("What kind of drink is this?", "vlm"),
    ("How many pens are on the table?", "vlm"),
]

LLM_PROMPT = (
    "You route questions for a voice assistant with a camera. Reply with exactly one word.\n"
    "text = no camera needed (facts, chat, memory, math).\n"
    "detections = which objects or people are present, how many, or where.\n"
    "vlm = needs a close look: colors, text, brands, comparisons, descriptions, what something is.\n"
    "Question: {q}\nRoute:"
)


def rule_route(question: str) -> str:
    route = reasoning.route_question(question, True, SCENE).route
    return route


def llm_route(client: httpx.Client, model: str, question: str, keep_alive: str) -> tuple[str, float]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    started = time.perf_counter()
    response = client.post(f"{base_url}/api/generate", json={
        "model": model, "stream": False, "keep_alive": keep_alive,
        "prompt": LLM_PROMPT.format(q=question),
        "options": {"temperature": 0, "num_predict": 4, "num_ctx": 512},
    })
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    word = (response.json().get("response") or "").strip().lower()
    for route in ("detections", "vlm", "text"):
        if route in word:
            return route, elapsed
    return f"?{word}", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", help="also evaluate this Ollama model as a classifier")
    parser.add_argument("--keep-alive", default="0")
    args = parser.parse_args()

    started = time.perf_counter()
    rule = [(q, want, rule_route(q)) for q, want in LABELED]
    rule_ms = (time.perf_counter() - started) * 1000 / len(LABELED)
    rule_ok = sum(want == got for _, want, got in rule)
    print(f"rule router: {rule_ok}/{len(LABELED)} correct, {rule_ms:.2f} ms per question")
    for q, want, got in rule:
        if want != got:
            print(f"  MISS {q!r}: expected {want}, got {got}")

    if args.llm:
        times, ok = [], 0
        with httpx.Client(timeout=300) as client:
            for q, want in LABELED:
                got, elapsed = llm_route(client, args.llm, q, args.keep_alive)
                times.append(elapsed)
                ok += got == want
                print(f"  {args.llm}: {elapsed:5.1f} s {'OK ' if got == want else 'BAD'} {q!r} -> {got}",
                      flush=True)
            if args.keep_alive != "0":
                client.post(
                    os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/generate",
                    json={"model": args.llm, "keep_alive": 0},
                )
        print(f"LLM router ({args.llm}, keep_alive={args.keep_alive}): {ok}/{len(LABELED)} correct, "
              f"median {statistics.median(times):.1f} s, first {times[0]:.1f} s, max {max(times):.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
