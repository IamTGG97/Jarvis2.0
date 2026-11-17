"""HTTP endpoints for listing, creating, deleting, and reading conversation sessions.

There is no authentication: anyone who can reach the backend can read or delete
every session. This is acceptable only for the local, single-user prototype.
"""

import logging
from datetime import datetime
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionSummary(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    turn_count: int
    preview: str | None


class SessionList(BaseModel):
    sessions: list[SessionSummary]


class SessionTurn(BaseModel):
    id: int
    transcript: str
    response: str
    created_at: datetime


class SessionTurns(BaseModel):
    session_id: UUID
    turns: list[SessionTurn]


async def _run_database(operation, *args):
    try:
        return await run_in_threadpool(operation, *args)
    except (psycopg.Error, RuntimeError) as error:
        logger.exception("Conversation session operation failed")
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is unavailable. Check PostgreSQL and DB_* settings.",
        ) from error


@router.get("", response_model=SessionList)
async def list_sessions(limit: int = 50) -> SessionList:
    limit = max(1, min(limit, 200))
    rows = await _run_database(database.list_sessions, limit)
    return SessionList(sessions=[SessionSummary(**row) for row in rows])


@router.post("", response_model=SessionSummary, status_code=201)
async def create_session() -> SessionSummary:
    row = await _run_database(database.create_session, uuid4())
    return SessionSummary(**row)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID) -> Response:
    deleted = await _run_database(database.delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=204)


@router.get("/{session_id}/turns", response_model=SessionTurns)
async def get_session_turns(session_id: UUID, limit: int = 200) -> SessionTurns:
    limit = max(1, min(limit, 1000))
    turns = await _run_database(database.get_session_turns, session_id, limit)
    if turns is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return SessionTurns(
        session_id=session_id,
        turns=[SessionTurn(**turn) for turn in turns],
    )
