import sqlite3
import uuid
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "incidents.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            error_text TEXT,
            service_name TEXT,
            root_cause TEXT,
            fix_suggestion TEXT,
            severity TEXT,
            confidence REAL,
            created_at TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id TEXT PRIMARY KEY,
            incident_id TEXT,
            latency REAL,
            confidence REAL,
            severity TEXT,
            service_name TEXT,
            created_at TEXT
        )
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at
        ON incidents(created_at)
        """)


def save_incident(incident: dict) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    incident["incident_id"],
                    incident.get("raw_input", ""),
                    incident.get("service_name", "unknown"),
                    incident.get("root_cause", ""),
                    incident.get("fix_suggestion", ""),
                    incident.get("severity", "Unknown"),
                    incident.get("confidence", 0.0),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception as e:
        logger.error("DB error: %s", e)


def get_all_error_texts():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM incidents").fetchall()
        return [dict(r) for r in rows]


def get_all_incidents():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        return {
            "total_incidents": total,
            "avg_confidence": 0,
            "avg_latency": 0,
            "by_severity": {}
        }
