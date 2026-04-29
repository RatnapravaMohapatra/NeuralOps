import re
import hashlib


# ─────────────────────────────
# INCIDENT ID
# ─────────────────────────────
def generate_incident_id(text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8].upper()
    return f"INC-{h}"


# ─────────────────────────────
# CONFIDENCE EVALUATION
# ─────────────────────────────
def evaluate_confidence(conf: float) -> str:
    if conf >= 0.8:
        return "High"
    elif conf >= 0.6:
        return "Medium"
    return "Low"


# ─────────────────────────────
# SERVICE KEYWORDS
# ─────────────────────────────
SERVICE_PATTERNS = {
    "payment": "payment-service",
    "predict": "predict-service",
    "auth": "auth-service",
    "user": "user-service",
    "order": "order-service",
    "cart": "cart-service",
    "inventory": "inventory-service",
    "recommendation": "recommendation-engine",
    "session": "session-service",
    "redis": "cache-service",
    "cache": "cache-service",
    "sql": "database-service",
    "database": "database-service",
    "kafka": "event-stream-service",
}


# ─────────────────────────────
# 🔥 SERVICE DETECTION (FINAL FIX)
# ─────────────────────────────
def infer_service_name(log: str, parsed: dict) -> str:
    log_lower = (log or "").lower()

    # ─────────────────────────────
    # 1. Use parser result if valid
    # ─────────────────────────────
    svc = parsed.get("service_name")
    if svc and svc != "unknown" and not svc.isdigit():
        return svc

    # ─────────────────────────────
    # 2. Explicit "Service: xyz"
    # ─────────────────────────────
    match = re.search(r"service[:=]\s*([\w-]+)", log_lower)
    if match:
        candidate = match.group(1)
        if not candidate.isdigit():
            return candidate

    # ─────────────────────────────
    # 3. API path inference (FIXED)
    # Only allow valid service-like names
    # ─────────────────────────────
    path_match = re.search(r"/([a-zA-Z][a-zA-Z0-9_-]+)", log_lower)
    if path_match:
        candidate = path_match.group(1)

        # 🚫 avoid numbers like /5
        if not candidate.isdigit() and len(candidate) > 2:
            return f"{candidate}-service"

    # ─────────────────────────────
    # 4. Keyword-based mapping
    # ─────────────────────────────
    for key, value in SERVICE_PATTERNS.items():
        if key in log_lower:
            return value

    # ─────────────────────────────
    # 5. Error-based inference
    # ─────────────────────────────
    if "timeout" in log_lower:
        return "network-service"

    if "oom" in log_lower or "memory" in log_lower:
        return "compute-service"

    if "kubernetes" in log_lower or "pod" in log_lower:
        return "k8s-platform"

    # ─────────────────────────────
    # 6. Final fallback (clean)
    # ─────────────────────────────
    return "core-platform-service"
