import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# =========================================================
# SYSTEM PROMPT (YOUR FULL INTELLIGENCE PRESERVED)
# =========================================================
SYSTEM_PROMPT = """You are a senior SRE root cause analysis agent with 10+ years of experience.

════════════════════════════════════════
THINKING PIPELINE (STRICT)
════════════════════════════════════════
1. Read the log carefully
2. Extract exact error signal
3. Match signal to table
4. Explain system-level failure
5. Assign confidence based on signal

════════════════════════════════════════
GOLDEN RULE
════════════════════════════════════════
"If it is not in the log, it is not in the answer."

NEVER guess. NEVER assume. NEVER hallucinate.

════════════════════════════════════════
ANTI-HALLUCINATION RULES
════════════════════════════════════════
- Do NOT assume DB issues without SQL/pool keywords
- Do NOT assume memory issues without heap/OOM keywords
- RAG is only a hint, NEVER override log

════════════════════════════════════════
SIGNAL → CAUSE → CONFIDENCE
════════════════════════════════════════

"pool exhausted" → DB connections fully used → 0.91  
"timed out" + URL → service slow/overloaded → 0.75  
"timed out" → internal blocking → 0.70  
"Connection refused" → service down → 0.85  
"500 Internal" → application bug → 0.80  
"OutOfMemoryError" → heap exhausted → 0.90  
"TLS / certificate" → SSL issue → 0.92  
"disk full" → storage full → 0.95  
"429 / rate limit" → API quota exceeded → 0.90  
"NullPointerException" → null reference bug → 0.80  

No match → 0.50

════════════════════════════════════════
ROOT CAUSE FORMAT (MANDATORY)
════════════════════════════════════════

Sentence 1: WHAT failed  
Sentence 2: EVIDENCE (quote exact log)  
Sentence 3: WHY (system explanation)

Example:
"The database connection pool is exhausted, preventing new requests from acquiring a connection.
This is evidenced by 'connection pool exhausted after 30s' in the log output.
This occurs when concurrent traffic exceeds pool size or slow queries hold connections too long."

════════════════════════════════════════
OUTPUT JSON ONLY
════════════════════════════════════════
{
  "root_cause": "3 sentences",
  "evidence": "exact log quote",
  "confidence": 0.0,
  "reasoning": "matched signal"
}
"""

# =========================================================
# USER PROMPT
# =========================================================
USER_PROMPT = """
LOG:
{summary}

Error Type: {error_type}
Service: {service_name}
Severity: {severity}

RAG (use only if directly relevant):
{rag_context}
"""

# =========================================================
# SAFETY FUNCTIONS
# =========================================================

def enforce_evidence(summary: str, evidence: str) -> str:
    if evidence and evidence in summary:
        return evidence
    return summary[:200]


def enforce_structure(text: str) -> str:
    parts = [s.strip() for s in text.split(".") if s.strip()]
    if len(parts) >= 3:
        return text
    return (
        f"{parts[0] if parts else 'Issue detected'}. "
        f"This is evidenced by the log output. "
        f"This occurs due to system-level failure conditions."
    )


def calibrate_confidence(evidence: str) -> float:
    e = evidence.lower()

    if "pool exhausted" in e:
        return 0.91
    if "connection refused" in e:
        return 0.85
    if "timeout" in e:
        return 0.75
    if "outofmemory" in e or "heap" in e:
        return 0.90
    if "disk" in e or "no space" in e:
        return 0.95
    if "429" in e:
        return 0.90

    return 0.5


# =========================================================
# BUILD AGENT
# =========================================================
def build_root_cause_agent(groq_api_key: str) -> callable:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # =========================================================
    # MAIN FUNCTION
    # =========================================================
    def analyze(parsed: dict, rag_results: list[dict]) -> dict:

        rag_context = "\n".join(
            f"- {r.get('root_cause', '')}"
            for r in rag_results
        ) or "No relevant incidents."

        try:
            result = chain.invoke({
                "summary": parsed.get("summary", ""),
                "error_type": parsed.get("error_type", ""),
                "service_name": parsed.get("service_name", ""),
                "severity": parsed.get("severity", ""),
                "rag_context": rag_context,
            })

        except Exception as e:
            logger.error("RootCauseAgent failed: %s", e)
            return {
                "root_cause": "Pipeline failure prevented analysis.",
                "evidence": parsed.get("summary", "")[:200],
                "confidence": 0.3,
                "reasoning": "LLM failure",
            }

        result = result or {}

        # =========================
        # 🔥 FINAL ENFORCEMENT
        # =========================
        root = result.get("root_cause", "Insufficient evidence.")

        evidence = enforce_evidence(
            parsed.get("summary", ""),
            result.get("evidence", "")
        )

        root = enforce_structure(root)

        confidence = calibrate_confidence(evidence)

        final = {
            "root_cause": root,
            "evidence": evidence,
            "confidence": confidence,
            "reasoning": result.get("reasoning", "Signal-based reasoning applied."),
        }

        logger.info(
            "RootCauseAgent: confidence=%.2f evidence=%s",
            final["confidence"],
            final["evidence"][:80],
        )

        return final

    return analyze
