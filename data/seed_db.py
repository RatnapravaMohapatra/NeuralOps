import sqlite3
import uuid
import logging

logger = logging.getLogger(__name__)

DB_PATH = "incidents.db"


# =========================================================
# CONNECTION
# =========================================================
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# INIT DB (RUN AT STARTUP)
# =========================================================
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # incidents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        service_name TEXT,
        severity TEXT,
        root_cause TEXT,
        fix_suggestion TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # feedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        incident_id TEXT,
        rating INTEGER,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ✅ INSERT SAMPLE DATA IF EMPTY (MAIN FIX)
    count = cursor.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    if count == 0:
        cursor.executemany("""
        INSERT INTO incidents (id, service_name, severity, root_cause, fix_suggestion, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            ("INC-001", "predict-service", "High", "Connection refused error", "Restart service", 0.85),
            ("INC-002", "order-service", "Critical", "Database connection pool exhausted", "Increase pool size", 0.90),
            ("INC-003", "recommendation-service", "High", "Out of memory error", "Increase heap size", 0.88)
        ])

        logger.info("Sample data inserted")

    conn.commit()
    conn.close()

    logger.info("Database initialized")


# =========================================================
# SAVE INCIDENT
# =========================================================
def save_incident(data: dict):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO incidents (id, service_name, severity, root_cause, fix_suggestion, confidence)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.get("incident_id"),
        data.get("service_name"),
        data.get("severity"),
        data.get("root_cause"),
        data.get("fix_suggestion"),
        data.get("confidence"),
    ))

    conn.commit()
    conn.close()


# =========================================================
# GET ALL INCIDENTS
# =========================================================
def get_all_incidents():
    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute(
        "SELECT * FROM incidents ORDER BY created_at DESC"
    ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


# =========================================================
# GET STATS
# =========================================================
def get_stats():
    conn = get_connection()
    cursor = conn.cursor()

    # total incidents
    total = cursor.execute(
        "SELECT COUNT(*) FROM incidents"
    ).fetchone()[0]

    # severity distribution
    severity_counts = cursor.execute("""
        SELECT severity, COUNT(*) as count
        FROM incidents
        GROUP BY severity
    """).fetchall()

    # average confidence
    avg_conf = cursor.execute("""
        SELECT AVG(confidence) FROM incidents
    """).fetchone()[0]

    conn.close()

    return {
        "total_incidents": total,
        "severity_distribution": {
            row[0]: row[1] for row in severity_counts
        },
        "avg_confidence": float(avg_conf) if avg_conf else 0.0,
        "avg_latency": 0.0  # placeholder (since not stored yet)
    }


# =========================================================
# SAVE FEEDBACK
# =========================================================
def save_feedback(incident_id: str, rating: int, comment: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO feedback (id, incident_id, rating, comment)
    VALUES (?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        incident_id,
        rating,
        comment
    ))

    conn.commit()
    conn.close()

    return {
        "status": "saved",
        "incident_id": incident_id,
        "rating": rating
    }


# =========================================================
# RAG SUPPORT
# =========================================================
def get_all_error_texts():
    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT root_cause, service_name, severity
        FROM incidents
    """).fetchall()

    conn.close()

    docs = []
    for row in rows:
        docs.append({
            "error_text": row["root_cause"] or "",
            "service_name": row["service_name"],
            "severity": row["severity"],
            "root_cause": row["root_cause"],
        })

    return docs
