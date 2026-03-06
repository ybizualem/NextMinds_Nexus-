"""
Database operations for the CTIC curriculum engine.

Uses psycopg3 (psycopg) for PostgreSQL + pgvector on Supabase.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Generator

import psycopg
from pgvector.psycopg import register_vector

from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DATABASE_URL
from src.db.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Get a database connection with pgvector registered."""
    if DB_HOST:
        conn = psycopg.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    else:
        conn = psycopg.connect(DATABASE_URL)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema():
    """Create tables and indexes if they don't exist."""
    logger.info("Initializing database schema...")
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    logger.info("Schema initialized successfully.")


def upsert_activity(
    conn: psycopg.Connection,
    activity_name: str,
    grade_band: str,
    stage: str,
    resource_url: str,
    resource_type: str,
    drive_id: str | None,
    embedding: list[float],
    description: str | None = None,
    estimated_time: int | None = None,
    keywords: list[str] | None = None,
) -> str:
    """
    Insert or update an activity. Returns 'inserted' or 'updated'.

    Uses ON CONFLICT on (resource_url, activity_name) to handle re-crawls.
    """
    result = conn.execute(
        """
        INSERT INTO activities (
            activity_name, grade_band, stage, resource_url, resource_type,
            drive_id, embedding, description, estimated_time_minutes, keywords,
            last_crawled_at, is_active
        ) VALUES (
            %(name)s, %(grade)s, %(stage)s, %(url)s, %(type)s,
            %(drive_id)s, %(embedding)s, %(desc)s, %(time)s, %(keywords)s,
            NOW(), TRUE
        )
        ON CONFLICT (resource_url, activity_name) DO UPDATE SET
            grade_band = EXCLUDED.grade_band,
            stage = EXCLUDED.stage,
            resource_type = EXCLUDED.resource_type,
            drive_id = EXCLUDED.drive_id,
            embedding = EXCLUDED.embedding,
            description = COALESCE(EXCLUDED.description, activities.description),
            estimated_time_minutes = COALESCE(EXCLUDED.estimated_time_minutes, activities.estimated_time_minutes),
            keywords = COALESCE(EXCLUDED.keywords, activities.keywords),
            last_crawled_at = NOW(),
            is_active = TRUE
        RETURNING (xmax = 0) AS is_insert
        """,
        {
            "name": activity_name,
            "grade": grade_band,
            "stage": stage,
            "url": resource_url,
            "type": resource_type,
            "drive_id": drive_id,
            "embedding": embedding,
            "desc": description,
            "time": estimated_time,
            "keywords": keywords,
        },
    )
    row = result.fetchone()
    return "inserted" if row[0] else "updated"


def mark_missing_inactive(conn: psycopg.Connection, active_urls: set[str]):
    """Mark activities not found in the latest crawl as inactive."""
    if not active_urls:
        return 0

    result = conn.execute(
        """
        UPDATE activities
        SET is_active = FALSE, updated_at = NOW()
        WHERE is_active = TRUE
        AND resource_url != ALL(%(urls)s)
        RETURNING id
        """,
        {"urls": list(active_urls)},
    )
    count = len(result.fetchall())
    if count > 0:
        logger.info(f"Marked {count} activities as inactive (no longer on site)")
    return count


def create_crawl_log(conn: psycopg.Connection, triggered_by: str = "manual") -> str:
    """Create a new crawl log entry. Returns the log ID."""
    result = conn.execute(
        """
        INSERT INTO crawl_logs (triggered_by, status)
        VALUES (%(by)s, 'running')
        RETURNING id
        """,
        {"by": triggered_by},
    )
    row = result.fetchone()
    return str(row[0])


def complete_crawl_log(
    conn: psycopg.Connection,
    log_id: str,
    added: int,
    updated: int,
    removed: int,
    errors: list[str] | None = None,
    status: str = "completed",
):
    """Update a crawl log entry when the crawl finishes."""
    conn.execute(
        """
        UPDATE crawl_logs SET
            completed_at = NOW(),
            activities_added = %(added)s,
            activities_updated = %(updated)s,
            activities_removed = %(removed)s,
            errors = %(errors)s,
            status = %(status)s
        WHERE id = %(id)s
        """,
        {
            "id": log_id,
            "added": added,
            "updated": updated,
            "removed": removed,
            "errors": json.dumps(errors) if errors else None,
            "status": status,
        },
    )


def search_activities(
    query_embedding: list[float],
    grade_band: str | None = None,
    stage: str | None = None,
    max_time: int | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Search activities using pgvector cosine similarity + metadata filters.

    Returns ranked results with similarity scores.
    """
    conditions = ["is_active = TRUE"]
    params: dict = {"embedding": query_embedding, "limit": limit}

    if grade_band:
        conditions.append("grade_band = %(grade)s")
        params["grade"] = grade_band
    if stage:
        conditions.append("stage ILIKE %(stage)s")
        params["stage"] = f"%{stage}%"
    if max_time:
        conditions.append("(estimated_time_minutes IS NULL OR estimated_time_minutes <= %(max_time)s)")
        params["max_time"] = max_time

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            id, activity_name, grade_band, stage, description,
            resource_url, resource_type, drive_id,
            estimated_time_minutes, keywords,
            1 - (embedding <=> %(embedding)s::vector) AS similarity
        FROM activities
        WHERE {where_clause}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(limit)s
    """

    with get_connection() as conn:
        result = conn.execute(query, params)
        columns = [desc.name for desc in result.description]
        rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def get_activity_stats() -> dict:
    """Get summary statistics about the activity catalog."""
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE is_active) AS active,
                COUNT(DISTINCT grade_band) AS grade_bands,
                COUNT(DISTINCT stage) AS stages,
                MIN(last_crawled_at) AS oldest_crawl,
                MAX(last_crawled_at) AS newest_crawl
            FROM activities
            """
        )
        row = result.fetchone()
        columns = [desc.name for desc in result.description]
        stats = dict(zip(columns, row))

        # Also get per-grade-band counts
        result = conn.execute(
            """
            SELECT grade_band, COUNT(*) AS count
            FROM activities
            WHERE is_active = TRUE
            GROUP BY grade_band
            ORDER BY grade_band
            """
        )
        stats["by_grade_band"] = {r[0]: r[1] for r in result.fetchall()}

        # Per-stage counts
        result = conn.execute(
            """
            SELECT stage, COUNT(*) AS count
            FROM activities
            WHERE is_active = TRUE
            GROUP BY stage
            ORDER BY stage
            """
        )
        stats["by_stage"] = {r[0]: r[1] for r in result.fetchall()}

    return stats


def update_health_status(resource_url: str, is_accessible: bool):
    """Update the health check timestamp and active status for an activity."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE activities SET
                last_verified_at = NOW(),
                is_active = CASE WHEN %(accessible)s THEN is_active ELSE FALSE END
            WHERE resource_url = %(url)s
            """,
            {"url": resource_url, "accessible": is_accessible},
        )
        conn.commit()
