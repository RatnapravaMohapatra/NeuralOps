import re
import hashlib


# ─────────────────────────────
# INCIDENT ID
# ─────────────────────────────
def generate_incident_id(text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8].upper()
    return f"INC-{h}"


# ─────────────────────────────
# CONFIDENCE EVALUATION (FIXED)
# ─────────────────────────────
def evaluate_confidence(conf: float) -> str:
    try:
        conf = float(conf)
    except:
        return "Low"

    if conf >= 0.75:
        return "High"
    elif conf >= 0.5:
        return "Medium"
    return "Low"


# ─────────────────────────────
# SERVICE KEYWORDS (EXPANDED)
# ─────────────────────────────
SERVICE_PATTERNS = {
    # business services
    "payment": "payment-service",
    "predict": "predict-service",
    "auth": "auth-service",
    "user": "user-service",
    "order": "order-service",
    "cart": "cart-service",
    "inventory": "inventory-service",
    "recommendation": "recommendation-engine",
    "session": "session-service",

    # infra / platform
    "redis": "cache-service",
    "cache": "cache-service",
    "sql": "database-service",
    "database": "database-service",
    "postgres": "database-service",
    "mysql": "database-service",
    "kafka": "event-stream-service",

    # gateway
    "gateway": "api-gateway",
    "nginx": "api-gateway",
    "ingress": "api-gateway",
}


# ─────────────────────────────
# 🔥 SERVICE DETECTION (STRONG VERSION)
# ─────────────────────────────
def infer_service_name(log: str, parsed: dict) -> str:
    log_lower = (log or "").lower()

    # ─────────────────────────────
    # 0. INFRA FIRST (VERY IMPORTANT)
    # ─────────────────────────────
    if "no space left" in log_lower or "disk" in log_lower:
        return "storage-service"

    if "insufficient cpu" in log_lower or "failedscheduling" in log_lower:
        return "k8s-platform"

    if "kubernetes" in log_lower or "pod" in log_lower:
        return "k8s-platform"

    if "oom" in log_lower or "memory" in log_lower:
        return "compute-service"

    if "timeout" in log_lower or "timed out" in log_lower:
        return "network-service"

    # ─────────────────────────────
    # 1. Use parser result (if valid)
    # ─────────────────────────────
    svc = parsed.get("service_name")

    if (
        svc
        and isinstance(svc, str)
        and svc.lower() not in ["unknown", "", "none"]
        and not svc.isdigit()
    ):
        return svc

    # ─────────────────────────────
    # 2. Explicit "service: xyz"
    # ─────────────────────────────
    match = re.search(r"service[:=]\s*([\w-]+)", log_lower)
    if match:
        candidate = match.group(1)
        if not candidate.isdigit() and len(candidate) > 2:
            return candidate

    # ─────────────────────────────
    # 3. Keyword mapping
    # ─────────────────────────────
    for key, value in SERVICE_PATTERNS.items():
        if key in log_lower:
            return value

    # ─────────────────────────────
    # 4. API path inference (SAFE)
    # ─────────────────────────────
    path_match = re.search(r"/([a-zA-Z][a-zA-Z0-9_-]{2,})", log_lower)
    if path_match:
        candidate = path_match.group(1)

        # avoid bad values like /api/v1 or /5
        if candidate not in ["api", "v1", "v2"] and not candidate.isdigit():
            return f"{candidate}-service"

    # ─────────────────────────────
    # 5. FINAL FALLBACK (NEVER UNKNOWN)
    # ─────────────────────────────
    return "core-platform-service"
