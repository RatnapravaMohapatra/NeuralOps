import sqlite3
import uuid
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "incidents.db")

SEED_INCIDENTS = [
    {
        "error_text": "SQLTimeoutException connection pool exhausted HikariPool getConnection",
        "service_name": "payment-service",
        "root_cause": "Database connection pool exhausted under sustained load spike. No circuit breaker in place.",
        "fix_suggestion": "Increase HikariCP pool size, add connection timeout config, implement circuit breaker pattern.",
        "severity": "Critical",
        "confidence": 0.91,
    },
    {
        "error_text": "OutOfMemoryError Java heap space Arrays copyOf InMemoryStore cache",
        "service_name": "recommendation-engine",
        "root_cause": "Unbounded in-memory cache growing without eviction policy, exhausting JVM heap.",
        "fix_suggestion": "Cap cache size with Caffeine/Guava eviction. Tune -Xmx JVM flag. Add heap alerting.",
        "severity": "Critical",
        "confidence": 0.88,
    },
    {
        "error_text": "pod eviction FailedScheduling Insufficient cpu node pressure kubernetes",
        "service_name": "api-gateway",
        "root_cause": "Node CPU exhaustion due to missing resource limits on pods, preventing new scheduling.",
        "fix_suggestion": "Set requests and limits on all pods. Enable cluster autoscaler. Add HPA for api-gateway.",
        "severity": "High",
        "confidence": 0.85,
    },
    {
        "error_text": "Redis maxmemory eviction allkeys-lru cache miss session store degraded",
        "service_name": "user-session-service",
        "root_cause": "Redis memory limit reached. LRU eviction invalidating active sessions causing miss storm.",
        "fix_suggestion": "Scale Redis cluster vertically. Implement tiered caching. Set maxmemory-policy to volatile-lru.",
        "severity": "High",
        "confidence": 0.83,
    },
    {
        "error_text": "NullPointerException service unavailable upstream timeout connection refused",
        "service_name": "notification-service",
        "root_cause": "Upstream dependency returned null response. No null guard or retry logic in calling code.",
        "fix_suggestion": "Add null checks and defensive programming. Implement retry with exponential backoff.",
        "severity": "Medium",
        "confidence": 0.78,
    },
    {
        "error_text": "disk full no space left on device log rotation failed",
        "service_name": "logging-agent",
        "root_cause": "Log rotation misconfigured. Logs accumulated and filled the disk partition.",
        "fix_suggestion": "Fix logrotate config. Add disk usage alerting at 70/85/95%. Mount dedicated log volume.",
        "severity": "High",
        "confidence": 0.87,
    },
    {
        "error_text": "SSL certificate expired TLS handshake failure connection reset",
        "service_name": "ingress-controller",
        "root_cause": "TLS certificate expired. No automated renewal (cert-manager) configured.",
        "fix_suggestion": "Install cert-manager with Let's Encrypt. Add certificate expiry monitoring alert.",
        "severity": "Critical",
        "confidence": 0.95,
    },
    {
        "error_text": "rate limit exceeded 429 too many requests throttled API quota",
        "service_name": "data-pipeline",
        "root_cause": "External API rate limit hit due to unbounded concurrent request fanout.",
        "fix_suggestion": "Implement token bucket rate limiter. Add request queuing. Cache upstream responses.",
        "severity": "Medium",
        "confidence": 0.80,
    },
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        if count == 0:
            logger.info("Seeding knowledge base with %d incidents...", len(SEED_INCIDENTS))
            for item in SEED_INCIDENTS:
                conn.execute(
                    """INSERT INTO incidents
                       (id, error_text, service_name, root_cause, fix_suggestion, severity, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        item["error_text"],
                        item["service_name"],
                        item["root_cause"],
                        item["fix_suggestion"],
                        item["severity"],
                        item["confidence"],
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            logger.info("Seed complete.")


def save_incident(incident: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO incidents
               (id, error_text, service_name, root_cause, fix_suggestion, severity, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
        conn.execute(
            """INSERT INTO request_log (id, incident_id, latency, confidence, severity, service_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                incident["incident_id"],
                incident.get("latency", 0.0),
                incident.get("confidence", 0.0),
                incident.get("severity", "Unknown"),
                incident.get("service_name", "unknown"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_all_incidents() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_error_texts() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, error_text, root_cause, fix_suggestion, service_name, severity FROM incidents"
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
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


def save_feedback(incident_id: str, rating: int, comment: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE incidents SET feedback_rating=?, feedback_comment=? WHERE id=?",
            (rating, comment, incident_id),
        )
