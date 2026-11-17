"""Optional memory retention: delete conversations that have been idle for
``MEMORY_RETENTION_DAYS`` days. Off by default (the variable is unset or 0).

The cleanup runs once at backend startup and then every
``MEMORY_RETENTION_CHECK_HOURS`` hours (default 6) while the backend runs.
"""

import asyncio
import logging
import os

import psycopg
from starlette.concurrency import run_in_threadpool

import database

logger = logging.getLogger(__name__)


def run_retention_once() -> int | None:
    """Purge expired sessions. Returns the number deleted, or None if off."""
    days = database.retention_days()
    if days is None:
        return None
    deleted = database.purge_expired_sessions(days)
    if deleted:
        logger.info("Retention removed %d conversation(s) idle for over %d days", deleted, days)
    return deleted


def retention_check_seconds() -> float:
    hours = float(os.getenv("MEMORY_RETENTION_CHECK_HOURS", "6") or 6)
    return max(hours, 0.01) * 3600


async def retention_loop() -> None:
    """Background task: purge now, then periodically. Failures are logged and
    retried on the next cycle; they never take the API down."""
    while True:
        try:
            await run_in_threadpool(run_retention_once)
        except (psycopg.Error, RuntimeError):
            logger.exception("Memory retention cleanup failed; will retry")
        await asyncio.sleep(retention_check_seconds())


def start_retention_task() -> asyncio.Task | None:
    """Start the retention loop if MEMORY_RETENTION_DAYS is set (validates it)."""
    days = database.retention_days()
    if days is None:
        return None
    logger.info("Memory retention is on: conversations idle for %d days are deleted", days)
    return asyncio.create_task(retention_loop(), name="memory-retention")
