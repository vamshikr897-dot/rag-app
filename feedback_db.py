import json
import sqlite3
import uuid
from datetime import datetime, timezone

import config


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(config.FEEDBACK_DB_PATH)


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TEXT,
                grade INTEGER,
                question TEXT,
                answer TEXT,
                response_category TEXT,
                sources_json TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                vote TEXT,
                feedback_reason TEXT,
                feedback_detail TEXT,
                regenerated_from_id TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON interactions(session_id)"
        )
        conn.commit()
    finally:
        conn.close()


def log_interaction(
    session_id: str,
    grade: int,
    question: str,
    answer: str,
    response_category: str | None,
    sources: list[dict],
    input_tokens: int,
    output_tokens: int,
    regenerated_from_id: str | None = None,
) -> str:
    interaction_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO interactions (
                id, session_id, created_at, grade, question, answer,
                response_category, sources_json, input_tokens, output_tokens,
                vote, feedback_reason, feedback_detail, regenerated_from_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
            """,
            (
                interaction_id,
                session_id,
                datetime.now(timezone.utc).isoformat(),
                grade,
                question,
                answer,
                response_category,
                json.dumps(sources),
                input_tokens,
                output_tokens,
                regenerated_from_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return interaction_id


def record_feedback(
    interaction_id: str, vote: str, reason: str | None = None, detail: str | None = None
) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE interactions SET vote = ?, feedback_reason = ?, feedback_detail = ? WHERE id = ?",
            (vote, reason, detail, interaction_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
