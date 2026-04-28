import sqlite3
import uuid
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "incidents.db")

CONF_MIN = 0.15


# ─────────────────────────────
# DB Connection
# ─────────────────────────────
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable WAL for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    return conn


# ─────────────────────────────
# Init DB
# ─────────────────────────────
def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                error_text TEXT NOT NULL,
                service_name TEXT,
                root_cause TEXT,
                fix_suggestion TEXT,
                severity TEXT,
                confidence REAL,
                feedback_rating INTEGER,
                feedback_comment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 🔥 Indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_log_incident_id ON request_log(incident_id)")

        logger.info("Database initialized")


# ─────────────────────────────
# Save Incident
# ─────────────────────────────
def save_incident(incident: Dict) -> None:
    try:
        with get_connection() as conn:
            confidence = max(CONF_MIN, float(incident.get("confidence", CONF_MIN)))

            conn.execute(
                """INSERT OR REPLACE INTO incidents
                   (id, error_text, service_name, root_cause, fix_suggestion, severity, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    incident["incident_id"],
                    incident.get("raw_input", "")[:2000],
                    incident.get("service_name", "unknown"),
                    incident.get("root_cause", ""),
                    incident.get("fix_suggestion", ""),
                    incident.get("severity", "Unknown"),
                    confidence,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.execute(
                """INSERT INTO request_log
                   (id, incident_id, latency, confidence, severity, service_name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    incident["incident_id"],
                    float(incident.get("latency", 0.0)),
                    confidence,
                    incident.get("severity", "Unknown"),
                    incident.get("service_name", "unknown"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    except Exception as e:
        logger.error("Failed to save incident: %s", e)


# ─────────────────────────────
# Fetch Data
# ─────────────────────────────
def get_all_incidents() -> List[Dict]:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Failed to fetch incidents: %s", e)
        return []


def get_all_error_texts() -> List[Dict]:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, error_text, root_cause, fix_suggestion, service_name, severity FROM incidents"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Failed to fetch error texts: %s", e)
        return []


# ─────────────────────────────
# Stats
# ─────────────────────────────
def get_stats() -> Dict:
    try:
        with get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM request_log").fetchone()[0]
            avg_lat = conn.execute("SELECT AVG(latency) FROM request_log").fetchone()[0]

            by_severity = conn.execute(
                "SELECT severity, COUNT(*) as count FROM incidents GROUP BY severity"
            ).fetchall()

            return {
                "total_incidents": total,
                "avg_confidence": round(avg_conf or 0, 3),
                "avg_latency": round(avg_lat or 0, 3),
                "by_severity": {row["severity"]: row["count"] for row in by_severity},
            }

    except Exception as e:
        logger.error("Failed to fetch stats: %s", e)
        return {}


# ─────────────────────────────
# Feedback
# ─────────────────────────────
def save_feedback(incident_id: str, rating: int, comment: str) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE incidents SET feedback_rating=?, feedback_comment=? WHERE id=?",
                (rating, comment, incident_id),
            )
    except Exception as e:
        logger.error("Failed to save feedback: %s", e)
