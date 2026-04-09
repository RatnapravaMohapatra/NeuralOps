"""
Shared utility tools available to all agents.
"""
import hashlib
import re
from datetime import datetime, timezone


def generate_incident_id(log_input: str) -> str:
    digest = hashlib.sha256(log_input.encode()).hexdigest()[:8].upper()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"INC-{ts}-{digest}"


def evaluate_confidence(confidence: float) -> str:
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"


def extract_service_hint(log_input: str) -> str | None:
    patterns = [
        r"service[:\s]+([a-z0-9_-]+)",
        r"app[:\s]+([a-z0-9_-]+)",
        r"pod[:\s]+([a-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_input, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def sanitize_log(log_input: str, max_length: int = 4000) -> str:
    log_input = log_input.strip()
    if len(log_input) > max_length:
        log_input = log_input[:max_length] + "\n[truncated]"
    return log_input
