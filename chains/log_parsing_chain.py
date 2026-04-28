import os
import logging
import time
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 2000
RETRY_ATTEMPTS = 2

VALID_SEVERITY = {"Critical", "High", "Medium", "Low"}


# ── Helpers ────────────────────────────────────────────
def _sanitize_log(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) > MAX_INPUT_LENGTH:
        return text[:MAX_INPUT_LENGTH] + " [truncated]"
    return text


def _safe_output(result: Any, fallback: str) -> Dict[str, str]:
    if not isinstance(result, dict):
        result = {}

    error_type = result.get("error_type") or "UnknownError"
    service_name = result.get("service_name") or "unknown"
    severity = result.get("severity") or "Medium"
    summary = result.get("summary") or fallback[:300]

    if severity not in VALID_SEVERITY:
        severity = "Medium"

    return {
        "error_type": error_type,
        "service_name": service_name,
        "severity": severity,
        "summary": summary,
    }


# ── Builder ────────────────────────────────────────────
def build_log_parsing_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0,
    )

    SYSTEM_PROMPT = """You are an expert SRE log parsing agent.

Extract structured information STRICTLY from the log.

Rules:
- Use ONLY what is present in the log (no guessing)
- If missing → use defaults
- Keep summary short and factual

Service name rules:
1. "Service: X" → X
2. API path like /predict → predict-service
3. Known patterns (payment, auth, api)
4. Else → "unknown"

Severity:
Critical / High / Medium / Low

Return ONLY JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "...",
  "summary": "..."
}
"""

    USER_PROMPT = """Analyze this log:

{log_input}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # ── Runner ─────────────────────────────────────────
    def parse_log(data: Dict[str, Any]) -> Dict[str, str]:
        start = time.time()

        raw_input = data.get("log_input", "")
        log_input = _sanitize_log(raw_input)

        if not log_input or len(log_input) < MIN_INPUT_LENGTH:
            raise ValueError("Invalid log input")

        last_error = None

        for attempt in range(RETRY_ATTEMPTS):
            try:
                result = chain.invoke({"log_input": log_input})
                safe = _safe_output(result, log_input)

                logger.info(
                    "LogParsing success (attempt=%d, latency=%.2fs)",
                    attempt + 1,
                    time.time() - start,
                )

                return safe

            except Exception as e:
                last_error = e
                logger.warning(
                    "LogParsing attempt %d failed: %s",
                    attempt + 1,
                    e,
                )
                time.sleep(0.8)

        # ── Fallback (never crash)
        logger.error("LogParsing failed after retries: %s", last_error)

        return {
            "error_type": "UnknownError",
            "service_name": "unknown",
            "severity": "Medium",
            "summary": log_input[:300],
        }

    return parse_log
