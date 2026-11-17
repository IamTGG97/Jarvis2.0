"""Compute embeddings for stored conversation turns that do not have one yet
(or that were embedded with a different model).

Usage (from the repository root, with Ollama running):

    .\\.venv\\Scripts\\python.exe backend\\backfill_embeddings.py
    .\\.venv\\Scripts\\python.exe backend\\backfill_embeddings.py --session <uuid> --batch-size 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill conversation-turn embeddings.")
    parser.add_argument("--session", type=UUID, help="only backfill this session")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()

    import database
    from embeddings import EmbeddingError

    database.initialize_database()
    started = time.perf_counter()
    try:
        updated = database.backfill_embeddings(
            session_id=args.session,
            batch_size=max(1, args.batch_size),
        )
    except EmbeddingError as error:
        print(f"Embedding failed, nothing more was updated: {error}", file=sys.stderr)
        return 1
    print(f"Embedded {updated} turn(s) in {time.perf_counter() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
