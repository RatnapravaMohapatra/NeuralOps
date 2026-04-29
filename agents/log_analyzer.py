import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.callbacks import CallbackManager

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}
MIN_INPUT_LENGTH = 5
MAX_INPUT_LENGTH = 2000


# ─────────────────────────────
# PROMPTS
# ─────────────────────────────
SYSTEM_PROMPT = """You are an expert SRE log analysis agent.

Extract facts ONLY from the log. No assumptions.

Return JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "Critical/High/Medium/Low",
  "summary": "..."
}
"""

USER_PROMPT = "Log:\n{log_input}"


# ─────────────────────────────
# 🔥 SEVERITY INFERENCE
# ─────────────────────────────
def _infer_severity(text: str) -> str:
    text = text.lower()

    if any(k in text for k in ["oomkilled", "outofmemory", "oom"]):
        return "Critical"

    if any(k in text for k in ["insufficient cpu", "failedscheduling"]):
        return "High"

    if any(k in text for k in ["timeout", "timed out"]):
        return "Medium"

    if any(k in text for k in ["warning", "deprecated"]):
        return "Low"

    return "Medium"


# ─────────────────────────────
# 🔥 SERVICE INFERENCE (NO UNKNOWN)
# ─────────────────────────────
def _infer_service(text: str) -> str:
    text = text.lower()

    if any(k in text for k in ["kubernetes", "pod", "node", "scheduler"]):
        return "k8s-platform"

    if any(k in text for k in ["payment", "transaction"]):
        return "payment-service"

    if any(k in text for k in ["auth", "token", "jwt"]):
        return "auth-service"

    if "order" in text:
        return "order-service"

    if "user" in text:
        return "user-service"

    if any(k in text for k in ["database", "sql", "postgres", "mysql"]):
        return "database-service"

    if any(k in text for k in ["redis", "cache"]):
        return "cache-service"

    if any(k in text for k in ["kafka", "queue", "stream"]):
        return "event-stream-service"

    if any(k in text for k in ["timeout", "latency", "network"]):
        return "network-service"

    if any(k in text for k in ["memory", "oom"]):
        return "compute-service"

    return "core-platform-service"  # 🔥 never unknown


# ─────────────────────────────
# 🔥 ERROR TYPE INFERENCE (NEW)
# ─────────────────────────────
def _infer_error_type(text: str) -> str:
    text = text.lower()

    if "timeout" in text:
        return "TimeoutError"

    if "connection refused" in text:
        return "ConnectionError"

    if "oom" in text or "memory" in text:
        return "MemoryError"

    if "cpu" in text:
        return "ResourceError"

    if "database" in text or "sql" in text:
        return "DatabaseError"

    return "UnknownError"


# ─────────────────────────────
# MAIN BUILDER
# ─────────────────────────────
def build_log_analyzer(api_key: str | None):

    # 🔥 Fallback mode (no API key)
    if not api_key:
        logger.warning("GROQ_API_KEY missing → using fallback analyzer")

        def fallback(input_data):
            log_input = (
                input_data.get("log_input", "")
                if isinstance(input_data, dict)
                else str(input_data)
            )

            return {
                "error_type": _infer_error_type(log_input),
                "service_name": _infer_service(log_input),
                "severity": _infer_severity(log_input),
                "summary": log_input[:300] or "No input",
            }

        return fallback

    # 🔥 LLM with LangSmith tracing
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
        callbacks=CallbackManager(),  # 🔥 REQUIRED
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    def analyze(input_data):
        log_input = (
            input_data.get("log_input", "")
            if isinstance(input_data, dict)
            else str(input_data)
        )

        # 🔥 Validation
        if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
            return {
                "error_type": "InvalidInput",
                "service_name": "core-platform-service",
                "severity": "Low",
                "summary": "Log input too short or invalid",
            }

        log_input = log_input[:MAX_INPUT_LENGTH]

        try:
            result = chain.invoke({"log_input": log_input})
        except Exception as e:
            logger.error("LogAnalyzer failed: %s", e)
            return {
                "error_type": _infer_error_type(log_input),
                "service_name": _infer_service(log_input),
                "severity": _infer_severity(log_input),
                "summary": log_input[:300],
            }

        if not isinstance(result, dict):
            result = {}

        # 🔥 Normalize everything
        severity = result.get("severity")
        if severity not in SEVERITY_LEVELS:
            severity = _infer_severity(log_input)

        service = result.get("service_name")
        if not service or service.lower() in ["unknown", ""]:
            service = _infer_service(log_input)

        error_type = result.get("error_type") or _infer_error_type(log_input)

        return {
            "error_type": error_type,
            "service_name": service,
            "severity": severity,
            "summary": result.get("summary") or log_input[:300],
        }

    return analyze
