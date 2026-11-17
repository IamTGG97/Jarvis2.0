"""Reproducible conversation-memory evaluation for Jarvis 2.0.

Seeds the conversation in ``memory_eval_fixtures.json`` into a dedicated eval
session in the real PostgreSQL database, asks exact-fact, paraphrased, and
never-mentioned questions, and reports:

* search recall@k: is the right earlier turn among the top k search results?
* context hit rate: is the right turn in the context actually sent to the LLM?
* optional end-to-end answer correctness through real Ollama (``--e2e``).

All rows it creates are deleted afterwards (and before seeding, in case an
earlier run was interrupted).

Usage (from the repository root):

    .\\.venv\\Scripts\\python.exe backend\\eval_memory.py
    .\\.venv\\Scripts\\python.exe backend\\eval_memory.py --e2e --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

FIXTURES_PATH = Path(__file__).with_name("memory_eval_fixtures.json")
EVAL_SESSION_ID = UUID("e7a1e7a1-0000-4000-8000-00000000e7a1")
DECOY_SESSION_ID = UUID("e7a1e7a1-0000-4000-8000-00000000dec0")
TURN_SPACING_MINUTES = 2
SEARCH_DEPTH = 10
RECALL_KS = (1, 3, 5)


@dataclass
class QuestionResult:
    id: str
    kind: str
    question: str
    search_rank: int | None = None
    noise_at_3: int = 0
    context_hit: bool | None = None
    context_turns: int = 0
    context_tokens: int = 0
    decoy_leak: bool = False
    retrieval_ms: float = 0.0
    answer: str | None = None
    answer_correct: bool | None = None
    admitted_missing: bool = False
    answer_ms: float | None = None
    notes: list[str] = field(default_factory=list)
    context: list[dict[str, str]] = field(default_factory=list, repr=False)


def load_fixtures(path: Path = FIXTURES_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def approx_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English)."""
    return max(1, round(len(text) / 4))


def context_token_estimate(context: list[dict[str, str]]) -> int:
    return sum(
        approx_tokens(turn["transcript"]) + approx_tokens(turn["response"])
        for turn in context
    )


def rank_of_target(
    ranked_turns: list[dict[str, str]],
    target_transcripts: set[str],
) -> int | None:
    """1-based rank of the first target turn in a ranked list, else None."""
    for index, turn in enumerate(ranked_turns, start=1):
        if turn["transcript"] in target_transcripts:
            return index
    return None


def _normalize(text: str) -> str:
    return text.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")


def admits_missing(answer: str, negative_markers: list[str]) -> bool:
    text = _normalize(answer)
    return any(marker in text for marker in negative_markers)


def score_answer(
    answer: str,
    question: dict,
    negative_markers: list[str],
    forbidden_terms: list[str],
) -> bool:
    """Keyword grading: positive answers must contain an expected term;
    negative answers must admit ignorance and must not invent a known fact."""
    text = _normalize(answer)
    if question["kind"] == "negative":
        admits = admits_missing(answer, negative_markers)
        invents = any(term in text for term in forbidden_terms)
        return admits and not invents
    return any(term in text for term in question["expect_any"])


def summarize(results: list[QuestionResult]) -> dict:
    positives = [result for result in results if result.kind != "negative"]
    summary: dict[str, object] = {"questions": len(results)}

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    for kind in ("all", "exact", "paraphrase"):
        group = [r for r in positives if kind == "all" or r.kind == kind]
        if not group:
            continue
        entry: dict[str, object] = {"n": len(group)}
        for k in RECALL_KS:
            entry[f"recall@{k}"] = rate(
                [r.search_rank is not None and r.search_rank <= k for r in group]
            )
        entry["mrr"] = round(
            statistics.mean(1 / r.search_rank if r.search_rank else 0 for r in group),
            3,
        )
        entry["context_hit"] = rate([bool(r.context_hit) for r in group])
        entry["avg_noise@3"] = round(statistics.mean(r.noise_at_3 for r in group), 2)
        answered = [r.answer_correct for r in group if r.answer_correct is not None]
        if answered:
            entry["answer_accuracy"] = rate(answered)
        summary[kind] = entry

    negatives = [r for r in results if r.kind == "negative"]
    if negatives:
        summary["negative_avg_noise@3"] = round(
            statistics.mean(r.noise_at_3 for r in negatives), 2
        )
    negative_answers = [r.answer_correct for r in negatives if r.answer_correct is not None]
    if negative_answers:
        summary["negative_answer_accuracy"] = rate(negative_answers)

    # When the needed turn is not in the context (never mentioned, or retrieval
    # missed it), the honest answer is "you haven't told me". Count how often the
    # model says so instead of guessing.
    unsupported = [
        r for r in results
        if r.answer is not None and (r.kind == "negative" or not r.context_hit)
    ]
    if unsupported:
        summary["unsupported_n"] = len(unsupported)
        summary["unsupported_admit_rate"] = rate(
            [r.admitted_missing for r in unsupported]
        )

    summary["decoy_leaks"] = sum(r.decoy_leak for r in results)
    summary["avg_context_turns"] = round(statistics.mean(r.context_turns for r in results), 2)
    summary["avg_context_tokens"] = round(statistics.mean(r.context_tokens for r in results))
    summary["max_context_tokens"] = max(r.context_tokens for r in results)
    summary["avg_retrieval_ms"] = round(statistics.mean(r.retrieval_ms for r in results), 1)
    answer_times = [r.answer_ms for r in results if r.answer_ms is not None]
    if answer_times:
        summary["avg_answer_ms"] = round(statistics.mean(answer_times))
    return summary


def cleanup(database) -> None:
    with database._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM conversation_sessions WHERE id = ANY(%s)",
            ([EVAL_SESSION_ID, DECOY_SESSION_ID],),
        )


def seed(database, fixtures: dict) -> None:
    sessions = (
        (EVAL_SESSION_ID, fixtures["turns"]),
        (DECOY_SESSION_ID, fixtures["decoy_session_turns"]),
    )
    with database._connect() as connection, connection.cursor() as cursor:
        for session_id, turns in sessions:
            cursor.execute(
                "INSERT INTO conversation_sessions (id) VALUES (%s)",
                (session_id,),
            )
            for index, turn in enumerate(turns):
                minutes_ago = (len(turns) - index) * TURN_SPACING_MINUTES
                cursor.execute(
                    """
                    INSERT INTO conversation_turns
                        (session_id, transcript, response, created_at)
                    VALUES (%s, %s, %s, now() - make_interval(mins => %s))
                    """,
                    (session_id, turn["transcript"], turn["response"], minutes_ago),
                )


def prepare_seeded_session(database) -> None:
    """Hook for derived data (for example embeddings) after seeding."""
    backfill = getattr(database, "backfill_embeddings", None)
    if backfill is not None:
        backfill(session_id=EVAL_SESSION_ID)
        backfill(session_id=DECOY_SESSION_ID)


def evaluate_retrieval(database, fixtures: dict) -> list[QuestionResult]:
    transcripts_by_key = {turn["key"]: turn["transcript"] for turn in fixtures["turns"]}
    decoy_transcripts = {turn["transcript"] for turn in fixtures["decoy_session_turns"]}
    results = []
    for question in fixtures["questions"]:
        result = QuestionResult(question["id"], question["kind"], question["question"])
        targets = {transcripts_by_key[key] for key in question["targets"]}

        ranked = database.search_conversation_turns(
            EVAL_SESSION_ID, question["question"], SEARCH_DEPTH
        )
        started = time.perf_counter()
        context = database.find_conversation_context(EVAL_SESSION_ID, question["question"])
        result.retrieval_ms = (time.perf_counter() - started) * 1000

        result.context_turns = len(context)
        result.context_tokens = context_token_estimate(context)
        context_transcripts = {turn["transcript"] for turn in context}
        result.decoy_leak = bool(context_transcripts & decoy_transcripts)
        result.noise_at_3 = sum(turn["transcript"] not in targets for turn in ranked[:3])
        if targets:
            result.search_rank = rank_of_target(ranked, targets)
            result.context_hit = bool(context_transcripts & targets)
        result.context = context
        results.append(result)
    return results


def evaluate_answers(fixtures: dict, results: list[QuestionResult]) -> None:
    from fastapi import HTTPException

    import main

    questions = {question["id"]: question for question in fixtures["questions"]}
    markers = fixtures["negative_markers"]
    forbidden = fixtures.get("negative_forbidden", {})
    for result in results:
        question = questions[result.id]
        started = time.perf_counter()
        try:
            answer = asyncio.run(
                main._generate_response(question["question"], result.context)
            )
        except HTTPException as error:
            result.notes.append(f"Ollama error {error.status_code}: {error.detail}")
            result.answer_correct = False
            continue
        finally:
            result.answer_ms = (time.perf_counter() - started) * 1000
        result.answer = answer
        result.admitted_missing = admits_missing(answer, markers)
        result.answer_correct = score_answer(
            answer, question, markers, forbidden.get(result.id, [])
        )
        print(
            f"  [{'ok' if result.answer_correct else 'MISS'}] {result.id}: "
            f"{answer[:110]!r} ({result.answer_ms / 1000:.1f}s)",
            flush=True,
        )


def print_report(results: list[QuestionResult], summary: dict) -> None:
    print("\nPer-question retrieval (rank = position in keyword/hybrid search):")
    for r in results:
        rank = "-" if r.search_rank is None else str(r.search_rank)
        hit = "n/a" if r.context_hit is None else ("hit" if r.context_hit else "MISS")
        print(
            f"  {r.id:<16} {r.kind:<10} rank={rank:<3} noise@3={r.noise_at_3} context={hit:<4} "
            f"turns={r.context_turns} ~tok={r.context_tokens}"
        )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--e2e", action="store_true", help="also grade real Ollama answers")
    parser.add_argument("--only", help="comma-separated question ids to run")
    parser.add_argument("--json", type=Path, help="write per-question results and summary here")
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
    import database

    fixtures = load_fixtures()
    if args.only:
        wanted = set(args.only.split(","))
        fixtures["questions"] = [q for q in fixtures["questions"] if q["id"] in wanted]

    database.initialize_database()
    cleanup(database)
    try:
        seed(database, fixtures)
        prepare_seeded_session(database)
        results = evaluate_retrieval(database, fixtures)
        if args.e2e:
            print("End-to-end answers through Ollama:")
            evaluate_answers(fixtures, results)
    finally:
        try:
            cleanup(database)
        except Exception as error:  # report, but keep the measured results
            print(
                f"WARNING: could not delete eval rows ({error}). Re-run the eval "
                "to clean up; it deletes leftovers before seeding.",
                file=sys.stderr,
            )

    summary = summarize(results)
    print_report(results, summary)
    if args.json:
        payload = {
            "summary": summary,
            "results": [
                asdict(r) for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
